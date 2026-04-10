from ml_pipeline import main
import os
import json
import itertools
import copy


# load base config
base_config = 'LstmBigRobustness'
experiment_path = os.path.join('experiments', base_config)
#experiment_path = os.path.join('experiments','LinRegTest2H')

model_config_path = experiment_path+os.sep+'model_config.json'
config = json.load(open(model_config_path))

#disabled_covariates = [[0,0,0], [0,0,1], [0,1,0], [0,1,1]]
disabled_covariates = [[0,0,0]]

context_limit = [456, 312, 96, 2304]#[96, 312, 456, 2304]
#context_limit = [96, 312, 888]
#context_limit = [None]

def safe_str(v):
	"""Return a filesystem-safe string for a hyperparameter value while preserving readability."""
	if v is None:
		return 'None'
	if isinstance(v, (list, tuple)):
		return '-'.join(safe_str(x) for x in v)
	return str(v).replace(os.sep, '_').replace(' ', '_')


def make_runs():
	# only sweep these two params; everything else comes from the loaded config
	param_grid = {
		'disabled_covariates': disabled_covariates,
		'context_limit': context_limit,
	}

	# short name mapping to keep folder names compact
	hp_name_map = {
		'disabled_covariates': 'dcov',
		'context_limit': 'fw',
	}

	base_experiment_dir = os.path.join('experiments', base_config)
	os.makedirs(base_experiment_dir, exist_ok=True)

	other_keys = [k for k in param_grid]
	other_values = [param_grid[k] for k in other_keys]

	run_idx = 0
	for combo in itertools.product(*other_values):
		run_idx += 1
		hp_values = dict(zip(other_keys, combo))

		cfg = copy.deepcopy(config)

		# apply the two hyperparameters into config
		cfg['run_config']['disabled_covariates'] = hp_values['disabled_covariates']
		# context_limit is normally in hp_config
		if 'hp_config' not in cfg:
			cfg['hp_config'] = {}
		cfg['hp_config']['context_limit'] = hp_values['context_limit']

		# build compact run name from only the parameters present in the grid
		parts = []
		for k in param_grid.keys():
			v = hp_values[k]
			short_key = hp_name_map.get(k, k)
			parts.append(f"{short_key}-{safe_str(v)}")

		run_name = f"run_" + "_".join(parts)
		cfg['run_config']['experiment_path'] = os.sep + os.path.join(base_experiment_dir, run_name)
		os.makedirs(cfg['run_config']['experiment_path'], exist_ok=True)
		with open(os.path.join(cfg['run_config']['experiment_path'], 'model_config.json'), 'w', encoding='utf-8') as f:
			json.dump(cfg, f, indent=2)

		print("Starting", run_name)
		try:
			main(cfg)
		except Exception as e:
			import traceback
			tb = traceback.format_exc()
			print(f"Run {run_name} failed with exception: {e}\n" + tb)
			# save traceback to experiment folder for inspection
			err_path = os.path.join(cfg['run_config']['experiment_path'], 'run_error.log')
			try:
				with open(err_path, 'w', encoding='utf-8') as ef:
					ef.write(tb)
			except Exception:
				# best-effort: don't crash the runner when saving the log
				pass


if __name__ == '__main__':
	make_runs()

