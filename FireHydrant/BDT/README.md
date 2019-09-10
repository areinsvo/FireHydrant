# FireHydrant/BDT

BDT training.

```bash
conda env update --file environment.yml
python xgbtrainer.py -h
```

## configuration for the trainer

The `xgbtrainer` use a configuration file encoded as YAML to specify some training hyperparameters.
It has following major groups:

```yml
training_fraction: 0.75,
hyperparam_seting:
  hyparam:
    default: 1
    range: [1, 10]
    loguniform: true
  ...
classifier_setting:
  ...
optimizer_setting:
  ...

```

- **training_fraction**: fraction of total dataset prepared for training.
- **hyperparam_setting**: options passed to XGBoost. see [XGBoost doc](https://xgboost.readthedocs.io/en/latest/parameter.html) for more info.
  Each entry is a dictionary with hyperparameter name as key, a dictionary with keys of `default`, `range` and `loguniform` as a value.
- **classifier_setting**: options for `XgboClassifier`.
- **optimizer_setting**: options for `XgboClassifier.optimize()`
