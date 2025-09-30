import json

# define the parameters and save them as a json file
my_params = {
    "architecture": "CNN",
    "learning_rate": 1e-03,
    "batch_size": 1024,
    "epochs": 120,
    "warmup_period": 20,
    "weight_decay" : 1e-5,
    "eta_min": 1e-6,
    "class_weights": [
        0.5,
        14.6
    ],
    "train_portion": 0.8,
    "random_seed": 42,
    "initialization": 'he',
    "use_wandb": True,
    "scheduler": True,
    "out_channel" : 16,
    "apply_early_stop": False,
    "stop_threshold": 5
}


# Save the dictionary to a JSON file
with open('CNNCV/parameters/my_params.json', 'w') as json_file:
    json.dump(my_params, json_file, indent=4)

 
print("Dictionary has been saved as 'data.json'")
