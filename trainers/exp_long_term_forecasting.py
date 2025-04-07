#from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
#from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from torchsummary import summary
import random
warnings.filterwarnings('ignore')
import io
import PIL
from torchvision.transforms import ToTensor

class PeriodicityReshape(nn.Module):
    def __init__(self, main_cycle):
        super(PeriodicityReshape, self).__init__()
        if main_cycle < 1:
            raise ValueError(f'Invalid main_cycle: {main_cycle}. Must be >= 1.')
        self.main_cycle = main_cycle
        
    def __assert_seq_len(self, x):
        _, n_steps, _ = x.shape
        seq_too_long = (n_steps % self.main_cycle) # is > 0 if n_steps is not a multiple of main_cycle -> True
        if seq_too_long:
            raise ValueError(f'''Number of steps {n_steps} is not a multiple of the main cycle ({n_steps}%{self.main_cycle}={n_steps%self.main_cycle}).
                             Suggested: Fill the sequence with zeros at the end to make it a multiple of the main cycle.''') 
            
    def apply(self, x, batch_size, n_features):
        self.__assert_seq_len(x)
        x = x.reshape(batch_size, -1, self.main_cycle, n_features).permute(0, 3, 1, 2)
        x = x.reshape(batch_size, -1, self.main_cycle).permute(0, 2, 1)
        return x

    def revert(self,x, batch_size, n_features):
        x = x.permute(0, 2, 1).reshape(batch_size, n_features, -1, self.main_cycle)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, n_features)
        return x

    def forward(self, x, n_features, direction):
        batch_size = x.shape[0]
        if direction == 'apply':
            return self.apply(x, batch_size, n_features)
        elif direction == 'revert':
            return self.revert(x, batch_size, n_features)
        else:
            raise ValueError(f'Invalid direction: {direction}. Use "apply" or "revert".')

from torch.optim.lr_scheduler import _LRScheduler
import math
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            print("Setting learning rate to:", lr)
            param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args, training_steps):
    if args.lradj == 'cosine_annealing_warmup':
        if not hasattr(adjust_learning_rate, 'scheduler'):
            total_lr_intervals = (training_steps * args.train_epochs) // args.lr_update_interval
            adjust_learning_rate.scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=int(total_lr_intervals*0.2),#args.first_cycle_steps,
                cycle_mult=1,#args.cycle_mult,
                max_lr=args.learning_rate,#args.max_lr,
                min_lr=0.0000000,#args.min_lr,
                warmup_steps=int(total_lr_intervals*0.03),#args.warmup_steps,
                gamma=1#args.gamma

            )
        adjust_learning_rate.scheduler.step(epoch)
    else:
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    for param_group in optimizer.param_groups:
        return param_group['lr']

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Save the model state dictionary
        torch.save(model.state_dict(), path)
        
        # Save the model summary
        summary_path = os.path.join(os.path.dirname(path), 'model_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary_str = summary(model, input_size=(3, 224, 224))  # Adjust input_size as per your model's requirement
            f.write(str(summary_str))
        
        self.val_loss_min = val_loss

class Args:
    def __init__(self, hp_config):
        #self.is_training = 0
        #self.model_id = 'test5'
        self.model = hp_config['architecture']
        self.freeze_encoder = hp_config['freeze_encoder']
        self.data = 'custom'
        #self.root_path = './data/electricity/'
        #self.data_path = 'electricity.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        #self.checkpoints = './checkpoints/'
        #if hp_config['history_of_days'] > 0:
        #    self.seq_len = hp_config['feature_window'] + hp_config['forecast_steps']#25#96
        #else:
        #    self.seq_len = hp_config['feature_window']
        self.history_of_days = hp_config['history_of_days']
        self.n_features = hp_config['n_features']
        self.seq_len = hp_config['feature_window'] 
        self.label_len = 48 # no longer needed in inverted Transformers
        self.pred_len = hp_config['feature_window']#hp_config['forecast_steps']#96
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = hp_config['token_size']#256#512 # Interesting to add to hp_config
        self.n_heads = 8 # Interesting to add to hp_config
        self.e_layers = hp_config['encoder_layers'] # Interesting to add to hp_config
        self.forecast_steps = hp_config['forecast_steps']
        self.forecast_horizons = hp_config['forecast_horizons']
        self.d_layers = 1 # Interesting to add to hp_config
        self.d_ff = 2048 # Interesting to add to hp_config
        self.moving_avg = 25 # Don't think this is used for iTransformer
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # Could be interesting to add to hp_config
        self.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned] Need to have the time encoding in the sam format
        self.activation = 'gelu' # Could be interesting to add to hp_config
        self.output_attention = False # Could be interesting to use this to see what the model pays attention to when making a prediction
        self.do_predict = False
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = hp_config['num_epochs']#10
        self.batch_size = hp_config['batch_size'] #32
        self.patience = 100
        self.learning_rate = hp_config['learning_rate']#0.0001
        self.des = 'test'
        self.loss = 'MSE'
        #self.lradj = 'type1'
        self.lradj = hp_config['lradj']
        self.lr_update_interval = hp_config['lr_update_interval']
        self.weight_decay = hp_config['weight_decay']
        self.use_amp = False
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.exp_name = 'MTSF'
        self.channel_independence = False
        self.inverse = False
        self.class_strategy = 'projection'
        #self.target_root_path = './data/electricity/'
        #self.target_data_path = 'electricity.csv'
        self.efficient_training = False
        self.use_norm = False#True
        self.partial_start_index = 0
        self.patch_size = hp_config['patch_size']
        self.mask_ratio = hp_config['mask_ratio']

        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self,hp_config, model_path, retrain_model=True):
        args = Args(hp_config)
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        #self.test_loader = test_loader
        self.model_path = model_path
        self.retrain_model = retrain_model
        self.criterion = nn.MSELoss()
        self.criterion_non_reduced = nn.MSELoss(reduction='none')
        #self.criterion_non_reduced = nn.L1Loss(reduction='none')
        self.criterion_forecast = nn.MSELoss()
        self.criterion_imputation = nn.MSELoss()
        self.criterion_alarm = nn.CrossEntropyLoss()
        self.criterion_uncertainty = nn.GaussianNLLLoss(reduction='none')
        self.periodicity_reshape = PeriodicityReshape(main_cycle=args.patch_size)
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == 'train':
            data_loader = self.train_loader 
        if flag == 'val':
            data_loader = self.val_loader
        #if flag == 'test':
        #    data_loader = self.test_loader
        data_set = None
        return data_set, data_loader
        data_set, data_loader = None, None #= data_provider(self.args, flag)
        from datahandler import DataHandler
        from dataprepper import DataPrepper
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, DataLoader
        data_handler = DataHandler('DataloaderOhio', "", dataset_name='Ohio2018')
        data_handler.load_data()
        prepper =  prepper = DataPrepper(data_handler.get_train_dataframes().keys(), data_handler, data_type="train", forecast_steps=6, scaler=StandardScaler(), fill_types=[-5,-5,-5,-5], experiment_path="experiments/test")
        if flag == 'train':
            prepper = DataPrepper(data_handler.get_train_dataframes().keys(), data_handler, data_type="train", forecast_steps=6, scaler=StandardScaler(), fill_types=[-5,-5,-5,-5], experiment_path="experiments/test")
        elif flag == 'val':
            prepper = DataPrepper(data_handler.get_test_dataframes().keys(), data_handler, data_type="test", forecast_steps=6, scaler=StandardScaler(), fill_types=[-5,-5,-5,-5], experiment_path="experiments/test")
        elif flag == 'test':
            prepper = DataPrepper(data_handler.get_test_dataframes().keys(), data_handler, data_type="test", forecast_steps=6, scaler=StandardScaler(), fill_types=[-5,-5,-5,-5], experiment_path="experiments/test")
        #prepper = DataPrepper(data_handler.get_train_dataframes().keys(), data_handler, data_type="train", forecast_steps=6, scaler=StandardScaler(), fill_types=[-5,-5,-5,-5], experiment_path="experiments/test", feature_list = ['cbg'])
        self.prepper = prepper
        self.scaler = prepper.scaler
        features_train, target_train = prepper.make_features_and_targetpair()
        none_tensor_x = torch.zeros(features_train.shape[0], dtype=torch.float32)
        none_tensor_y = torch.zeros(features_train.shape[0], dtype=torch.float32)
        class CustomTensorDataset(TensorDataset):
            def __init__(self, *tensors, scale=None):
                super().__init__(*tensors)
                self.scale = scale
        train_data = CustomTensorDataset(features_train, target_train, none_tensor_x, none_tensor_y)
        train_data.scale = True
        data_loader = DataLoader(train_data, shuffle=True, batch_size=32)
        data_set = train_data
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def calculate_loss(self, outputs, batch_x, batch_y, outputs_var=None, tb_writer=None, iter_count=None):
        """
        Calculates the loss using either MSE or GaussianNLLLoss depending on whether variance predictions are provided.
        
        Args:
            outputs (torch.Tensor): The model outputs (mean predictions) of shape [Batch, Time, Variate].
            batch_x (torch.Tensor): The input tensor of shape [Batch, Time, Variate].
            batch_y (torch.Tensor): The target tensor of shape [Batch, Time, Variate].
            outputs_var (torch.Tensor, optional): The model's variance predictions of shape [Batch, Time, Variate].
            tb_writer (SummaryWriter, optional): TensorBoard writer for logging.
            iter_count (int, optional): Current iteration count for logging.
            
        Returns:
            torch.Tensor: The masked loss.
        """
        # Create the difference mask
        difference_mask = (batch_x != batch_y).float()
        
        # Create the valid mask where batch_y is not -8 or -9
        valid_mask = ((batch_y != -8) & (batch_y != -9)).float()
        
        # Combine the masks
        combined_mask = difference_mask * valid_mask
        
        if outputs_var is None:
            # Use MSE loss if no variance predictions are provided
            loss = self.criterion_non_reduced(outputs, batch_y)
        else:
            # Use GaussianNLL loss when variance predictions are provided
            # Ensure variance is positive
            outputs_var = torch.clamp(outputs_var, min=1e-6)
            loss = self.criterion_uncertainty(outputs, batch_y, outputs_var)
        
        # Apply the combined mask while maintaining gradients
        masked_loss = loss * combined_mask.to(outputs.device)
        
        # Normalize the masked loss
        if combined_mask.sum() == 0:
            # Return a differentiable zero tensor
            masked_loss = torch.zeros(1, requires_grad=True, device=outputs.device)
        else:
            masked_loss = masked_loss.sum() / (combined_mask.sum() + 1e-6)  # Add small epsilon to avoid division by zero
      
        if iter_count is not None and iter_count % 5000 == 0:  # Plotting condition
            # Plotting
            batch_index = 15  # Select the first batch for visualization
            # Replace -8 and -9 with NaN
            batch_y = batch_y.clone()
            batch_y[(batch_y == -8) | (batch_y == -9) | (batch_y == -6)] = np.nan

            batch_x = batch_x.clone()
            batch_x[(batch_x == -8) | (batch_x == -9) | (batch_x == -6)] = np.nan

            outputs = outputs.clone()
            outputs[(outputs == -8) | (outputs == -9) | (outputs == -6)] = np.nan
            # Determine the color scale limits using masks to ignore NaN values
            vmin = min(
                torch.quantile(batch_x[~torch.isnan(batch_x)], 0.05).item(),
                torch.quantile(batch_y[~torch.isnan(batch_y)], 0.05).item(),
                torch.quantile(outputs[~torch.isnan(outputs)], 0.05).item()
            )
            vmax = max(
                torch.quantile(batch_x[~torch.isnan(batch_x)], 0.95).item(),
                torch.quantile(batch_y[~torch.isnan(batch_y)], 0.95).item(),
                torch.quantile(outputs[~torch.isnan(outputs)], 0.95).item()
            )
            

            plt.figure(figsize=(20, 5))
            
            plt.subplot(1, 4, 1)
            plt.imshow(combined_mask[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='gray')
            plt.title('Mask')
            plt.colorbar()
            
            plt.subplot(1, 4, 2)
            plt.imshow(batch_y[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title('Batch Y')
            plt.colorbar()
            
            plt.subplot(1, 4, 3)
            plt.imshow(batch_x[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title('Batch X')
            plt.colorbar()
            
            plt.subplot(1, 4, 4)
            plt.imshow(outputs[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title('Outputs')
            plt.colorbar()
            
            plt.tight_layout()
            #plt.show()
            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Convert the buffer to a tensor
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            
            # Add the image to TensorBoard
            if tb_writer is not None:
                tb_writer.add_image('Outputs', image, iter_count)
        return masked_loss

    def calculate_test_target_loss(self, outputs, batch_y, test_target, forecast_steps=0):
        loss = self.criterion(outputs[:,-forecast_steps:,test_target], batch_y[:,-forecast_steps:,test_target])
        return loss

    def calculate_loss_prev(self, outputs, batch_x, batch_y, masked_tokens=None):
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        
        # Create a mask where batch_y is not -8
        valid_mask = batch_y != -8  # Shape: [Batch, Time, Variate]
        
        if masked_tokens is not None:
            # Expand masked_tokens to match the dimensions of outputs and batch_y
            masked_tokens = masked_tokens.unsqueeze(1).expand(-1, batch_y.size(1), -1)  # Shape: [Batch, Time, Variate]
            # Combine masks
            mask = valid_mask & masked_tokens
        else:
            mask = valid_mask  # Use valid_mask if masked_tokens is None
        
        # Compute the loss without reduction
        loss = self.criterion_non_reduced(outputs, batch_y)  # Shape: [Batch, Time, Variate]
        
        # Apply the mask
        masked_loss = loss * mask.float()
        
        # Ensure that we are not dividing by zero
        if mask.sum() == 0:
            masked_loss = torch.tensor(0.0, device=self.device)
        else:
            masked_loss = masked_loss.sum() / mask.sum()
        
        if False:#random.randint(1, 100) == 1:
            # Plotting
            
            batch_index = 15  # Select the first batch for visualization
            time_index = 0   # Select the first time step for visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(mask[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='gray')
            plt.title('Mask')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(batch_y[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='viridis')
            plt.title('Batch Y')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(outputs[batch_index, :, :].cpu().detach().numpy(), aspect='auto', cmap='plasma')
            plt.title('outputs')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()
        
        return masked_loss

    def calculate_regression_metrics(self, pred, targets):
        regression_metrics = {}
        for forecast_horizon in self.args.forecast_horizons:
            pred_step = pred[:, forecast_horizon, :].flatten()
            targets_step = targets[:, forecast_horizon, :].flatten()
            mae = metrics.mean_absolute_error(targets_step, pred_step)
            mse = metrics.mean_squared_error(targets_step, pred_step)
            rmse = np.sqrt(mse)
            mape = metrics.mean_absolute_percentage_error(targets_step, pred_step)
            #mspe = metrics.mean_squared_log_error(targets_step, pred_step)
            regression_metrics["forecast_horizon_"+str(forecast_horizon)] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}
            
        print(regression_metrics)
        
    def calculate_classification_metrics(pred, targets):
        classification_metrics = {}
        pass
    def calculate_uncertainty_metrics(pred, uncertainty, targets):
        pass


    def vali(self, vali_data, vali_loader, tb_writer=None):
        total_loss = []
        total_loss_targets = []
        self.model.eval()
        with torch.no_grad():
            batch_x_mark, batch_y_mark = None, None
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'custom' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x = self.periodicity_reshape(batch_x, self.args.n_features, 'apply')
                batch_y = self.periodicity_reshape(batch_y, self.args.n_features, 'apply')
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    #if self.args.output_attention:
                    #    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    #else:
                    outputs, outputs_var = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_variance=True)
                loss = self.calculate_loss(outputs, batch_x, batch_y, outputs_var=outputs_var, tb_writer=tb_writer)
                #loss_target = self.calculate_test_target_loss(outputs, batch_y, vali_loader.dataset.test_target_index, forecast_steps=self.args.forecast_steps)
                loss_target = self.calculate_test_target_loss(outputs, batch_y, 0, forecast_steps=self.args.forecast_steps)
                total_loss.append(loss.item())
                total_loss_targets.append(loss_target.item())
        total_loss = np.average(total_loss)
        total_loss_targets = np.average(total_loss_targets)
        self.model.train()
        return total_loss, total_loss_targets

    def train(self, train_loader, vali_loader, return_variance=True):
        #train_data, train_loader = self._get_data(flag='train')
        tb_dir = os.path.dirname(self.model_path)
        log_dir = os.path.join(tb_dir, 'logs', time.strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        #vali_data, vali_loader = self._get_data(flag='val')
        vali_data = None
        #test_data, test_loader = self._get_data(flag='test')

        path = self.model_path #os.path.join(self.args.checkpoints, setting)
        dir_name = os.path.dirname(path)  # get the directory name

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.retrain_model == False:
            self.model.load_state_dict(torch.load(self.model_path))
            if self.args.freeze_encoder:
                self.model.freeze_encoder()
            print('Model loaded from:', self.model_path)
            # Validate baseline performance of the model
            vali_loss, _ = self.vali(vali_data, vali_loader)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                return self.model
        self.model.tb_writer = writer
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        total_iters = 0
        lr_intervals = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_targets = []

            self.model.train()
            epoch_time = time.time()
            #This needs to be added back to support timestamps
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x_mark, batch_y_mark = None, None
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                total_iters += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                batch_x = self.periodicity_reshape(batch_x, self.args.n_features, 'apply')
                batch_y = self.periodicity_reshape(batch_y, self.args.n_features, 'apply')

                #batch_x[:, -2:, :] = -7
                #states = torch.zeros_like(batch_y) # Hypoglycemia
                #states[batch_y > 70] = 1 # Normal
                #states[batch_y > 180] = 2 # Hyperglycemia
                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'custom' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                #if self.args.output_attention:
                #    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #else:
                #    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # If doing masked autoencoding
                outputs, outputs_var = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_variance=return_variance)

                loss = self.calculate_loss(outputs, batch_x, batch_y, outputs_var=outputs_var,tb_writer=writer, iter_count=total_iters)
                #train_loss_target = self.calculate_test_target_loss(outputs, batch_y, train_loader.dataset.test_target_index, forecast_steps=self.args.forecast_steps)
                train_loss_target = self.calculate_test_target_loss(outputs, batch_y, 0, forecast_steps=self.args.forecast_steps)

                train_loss.append(loss.item())
                train_loss_targets.append(train_loss_target.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    iter_count = 0
                    time_now = time.time()
                    writer.add_scalar('Loss/train_iters', loss.item(), epoch * train_steps + i)
                    writer.add_scalar('Loss/train_target_iters', train_loss_target.item(), epoch * train_steps + i)
                
                if total_iters % self.args.lr_update_interval == 0:

                    vali_loss, vali_loss_target = self.vali(vali_data, vali_loader)
                    print("\titers: {0}, epoch: {1} | val_loss: {2:.7f}".format(i + 1, epoch + 1, vali_loss))
                    time_now = time.time()

                    #vali_loss = self.vali(vali_data, vali_loader, criterion)
                    writer.add_scalar('Loss/val_iters', vali_loss, epoch * train_steps + i)
                    writer.add_scalar('Loss/val_target_iters', vali_loss_target, epoch * train_steps + i)
                    lr_intervals += 1
                    lr = adjust_learning_rate(model_optim, lr_intervals, self.args, train_steps)
                    writer.add_scalar('LearningRate', lr, epoch * train_steps + i)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_targets = np.average(train_loss_targets)
            #writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/train_target', train_loss_targets, epoch)
            vali_loss, vali_loss_target = self.vali(vali_data, vali_loader)
            writer.add_scalar('Loss/val', vali_loss, epoch)
            writer.add_scalar('Loss/val_target', vali_loss_target, epoch)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, 0))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = self.model_path #path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        writer.close()
        return self.model

    def test(self, test_loader, test=1, scaler=None):
        ii = 0
        #test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(self.model_path))

        preds = []
        trues = []
        stds = []

        self.model.eval()
        all_forecast = []
        all_imputation = []
        all_alarm = []
        all_uncertainty_forecast = []
        all_uncertainty_imputation = []
        all_batch_y = []
        all_batch_x = []
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x_mark, batch_y_mark = None, None
            for i, (batch_x, batch_y, ) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'custom' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x = self.periodicity_reshape(batch_x, self.args.n_features, 'apply')
                batch_y = self.periodicity_reshape(batch_y, self.args.n_features, 'apply')
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, outputs_var = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_variance=True)
                else:
                    #if self.args.output_attention:
                    outputs, outputs_var = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_variance=True)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs_var = outputs_var[:, -self.args.pred_len:, f_dim:]
                outputs_std = torch.sqrt(outputs_var)
                batch_x = self.periodicity_reshape(batch_x, self.args.n_features, 'revert')
                batch_y = self.periodicity_reshape(batch_y, self.args.n_features, 'revert')
                outputs = self.periodicity_reshape(outputs, self.args.n_features, 'revert')
                outputs_std = self.periodicity_reshape(outputs_std, self.args.n_features, 'revert')
                
                outputs = outputs.detach().cpu().numpy()
                outputs_std = outputs_std.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_y = np.where(np.isin(batch_y, [-8, -9]), np.nan, batch_y)

                # Store pre-transform values
                outputs_pre = outputs.copy()
                batch_y_pre = batch_y.copy()

                # Perform transforms
                pred = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
                true = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)
                std = outputs_std * scaler.scale_[None, None, :] 

                preds.append(pred[:,-self.args.forecast_steps:,:])
                trues.append(true[:,-self.args.forecast_steps:,:])
                stds.append(std[:,-self.args.forecast_steps:,:])

        preds = np.array(preds[:-1])
        trues = np.array(trues[:-1])
        stds = np.array(stds[:-1])  # Convert stds to a numpy array
        #print('test shape:', preds.shape, trues.shape, stds.shape)  # Include stds shape in the print statement
        # Check if trues is empty
        if trues.size == 0:
            print("Warning: 'trues' is empty. Skipping reshaping.")
        else:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            stds = stds.reshape(-1, stds.shape[-2], stds.shape[-1])

        return preds, trues, stds


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            #path = os.path.join(self.args.checkpoints, setting)
            #best_model_path = self.model_path#path + '/' + 'checkpoint.pth'
            #self.model.load_state_dict(torch.load(best_model_path))
            self.model.load_state_dict(torch.load(self.model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)

        #np.save(folder_path + 'real_prediction.npy', preds)

        return