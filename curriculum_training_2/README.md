# Curriculum training (model-defined)

To run this experiment:

```playpen curriculum_training_2/lora_trainer_curriculum.py -l llama3-8b```

Please make sure to check your training configuration in ```training_config.yaml``` - note in particular the directory for the failed base model instances.

To create ```failed_instances.json``` of the model, take a look in the ```failure_driven_learning``` experiment.
