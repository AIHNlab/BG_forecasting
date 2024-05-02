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

warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
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
        elif score < self.best_score + self.delta:
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
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class Args:
    def __init__(self, hp_config):
        #self.is_training = 0
        #self.model_id = 'test5'
        self.model = hp_config['architecture']
        self.data = 'custom'
        #self.root_path = './data/electricity/'
        #self.data_path = 'electricity.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        #self.checkpoints = './checkpoints/'
        self.seq_len = hp_config['feature_window']#25#96
        self.label_len = 48 # no longer needed in inverted Transformers
        self.pred_len = hp_config['forecast_window']#96
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512 # Interesting to add to hp_config
        self.n_heads = 8 # Interesting to add to hp_config
        self.e_layers = 2 # Interesting to add to hp_config
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
        self.patience = 3
        self.learning_rate = hp_config['learning_rate']#0.0001
        self.des = 'test'
        self.loss = 'MSE'
        self.lradj = 'type1'
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
        self.use_norm = True
        self.partial_start_index = 0

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            batch_x_mark, batch_y_mark = None, None
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'custom' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
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
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_loader, vali_loader):
        #train_data, train_loader = self._get_data(flag='train')
        #vali_data, vali_loader = self._get_data(flag='val')
        vali_data = None
        #test_data, test_loader = self._get_data(flag='test')

        path = self.model_path #os.path.join(self.args.checkpoints, setting)
        dir_name = os.path.dirname(path)  # get the directory name

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if self.retrain_model == False:
            self.model.load_state_dict(torch.load(self.model_path))
            print('Model loaded from:', self.model_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #This needs to be added back to support timestamps
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x_mark, batch_y_mark = None, None
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'custom' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, 0))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = self.model_path #path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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

        self.model.eval()
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                #if test_data.scale and self.args.inverse:
                #if scaler:
                #    shape = outputs.shape
                #    outputs = scaler.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #    batch_y = scaler.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                #pred =  scaler.inverse_transform(outputs)
                #true =  scaler.inverse_transform(batch_y)
                pred = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
                true = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)

                #plt.plot(scaler.inverse_transform(pred[i]), label='Predicted', linestyle='dashed')
                #plt.plot(scaler.inverse_transform(true[i]), label='True')
                #plt.ylim(0, 300)
                #plt.legend()
                #plt.show()

                preds.append(pred)
                trues.append(true)
                #if i % 20 == 0:
                #    input = batch_x.detach().cpu().numpy()
                #    if test_data.scale and self.args.inverse:
                #        shape = input.shape
                #        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        print("yo")
        preds = np.array(preds[:-1])
        trues = np.array(trues[:-1])
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        #print('mse:{}, mae:{}'.format(mse, mae))
        #f = open("result_long_term_forecast.txt", 'a')
        #f.write(setting + "  \n")
        #f.write('mse:{}, mae:{}'.format(mse, mae))
        #f.write('\n')
        #f.write('\n')
        #f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)
        # Assuming `preds` and `trues` are your lists of predictions and true values

        return preds, trues


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