# Failure-driven learning

To run this experiment, do:

```playpen learn-from-failures --learner {model} --teacher {model(s)}```

Please make sure that the cli.py used by your playpen installation is the one from this repo: ```(playpen/cli.py)```, otherwise this experiment will not work (with data_utils.py also in that folder - seems like semantically placing it there makes more sense?).

Also, please check your training configuration in ```training_config.yaml``` before running this experiment.
