import configparser

model_training_config = configparser.ConfigParser()
model_init_config = configparser.ConfigParser()

model_training_config.read("config/training/model_training.ini")
model_init_config.read("config/init/model_init.ini")

# For Model Training
training_block_size = int(model_training_config["ModelTrainingConfig"]["block_size"])
training_d_model = int(model_training_config["ModelTrainingConfig"]["d_model"])
training_n_layer = int(model_training_config["ModelTrainingConfig"]["n_layer"])
training_n_head = int(model_training_config["ModelTrainingConfig"]["n_head"])
training_d_ff = int(model_training_config["ModelTrainingConfig"]["d_ff"])
training_dropout = float(model_training_config["ModelTrainingConfig"]["dropout"])
training_batch_size = int(model_training_config["ModelTrainingConfig"]["batch_size"])
training_grad_accum_steps = int(model_training_config["ModelTrainingConfig"]["grad_accum_steps"])
training_epochs = int(model_training_config["ModelTrainingConfig"]["epochs"])
training_max_lr = float(model_training_config["ModelTrainingConfig"]["max_lr"])
training_min_lr = float(model_training_config["ModelTrainingConfig"]["min_lr"])
training_warmup_ratio = float(model_training_config["ModelTrainingConfig"]["warmup_ratio"])
training_weight_decay = float(model_training_config["ModelTrainingConfig"]["weight_decay"])
training_grad_clip = float(model_training_config["ModelTrainingConfig"]["grad_clip"])
training_label_smoothing = float(model_training_config["ModelTrainingConfig"]["label_smoothing"])
training_max_new_tokens = int(model_training_config["AfterTrainingConfig"]["max_new_tokens"])
training_temperature = float(model_training_config["AfterTrainingConfig"]["temperature"])
training_top_k = int(model_training_config["AfterTrainingConfig"]["top_k"])

# For Model Initialization
init_max_new_tokens = int(model_init_config["ModelInitConfig"]["max_new_tokens"])
init_temperature = float(model_init_config["ModelInitConfig"]["temperature"])
init_top_k = int(model_init_config["ModelInitConfig"]["top_k"])