import json
import math

# define the parameters and save them as a json file
my_params = {
    "architecture": "VITCV",
    "batch_size": 1024,
    "epochs": 100,
    "warmup_period": 20,
    "learning_rate": 1e-3,
    "eta_min": 1e-6,
    "weight_decay" : 1e-5,
    "class_weights": [0.5, 14.6],
    "use_wandb": True,
    "scheduler": True,
    "img_W" : 105,
    "img_H" : 113,
    "kernel_size" : 7,
    "drop_out_stoch" :0.3,
    "drop_out_att" :0.2,
    "embedding_dim" : 64,
    "num_heads" : 8,
    "num_blks" :2,
    "apply_early_stop": False,
    "stop_threshold": 5,
    "random_seed": 42,
    "train_portion" : 0.8,
}


# Save the dictionary to a JSON file
with open('VITCV/parameters/my_params.json', 'w') as json_file:
    json.dump(my_params, json_file, indent=4)

print("Dictionary has been saved as 'data.json'")
