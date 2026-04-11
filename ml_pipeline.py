"""
Backwards-compatibility shim. Canonical location: pipeline/

All functionality has been moved to the pipeline package:
  - pipeline.orchestrator: main(), init_experiment_directory()
  - pipeline.training: train_model(), create_temporal_split(), train_model_glucobench()
  - pipeline.evaluation: evaluate_model(), evaluate_model_glucobench(), plot_per_horizon()
"""
import os
import json

from pipeline.orchestrator import main, init_experiment_directory
from pipeline.training import train_model, create_temporal_split, train_model_glucobench
from pipeline.evaluation import evaluate_model, evaluate_model_glucobench, plot_per_horizon

if __name__ == "__main__":
    experiment_path = os.path.join('experiments','NewMainValidation')

    model_config_path = experiment_path+os.sep+'model_config.json'
    config = json.load(open(model_config_path))
    main(config)
