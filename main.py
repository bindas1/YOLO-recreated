import wandb
import train
import evaluation
import dataset
import utils


# from the paper for SGD
MOMENTUM = 0.9
DECAY = 0.0005
# model is trained for 135 epochs
# first 5 epochs from 0.001 to 0.1
# 75 epochs 0.01 epochs
# 30 epochs 0.001
# 30 epochs 0.0001

LEARNING_RATE = 0.000005
EPOCHS = 80

if __name__ == "__main__":
	wandb.login()

	config = dict(
	    # TRAINING PARAMS AND HYPERPARAMS
	    epochs=EPOCHS,
	    learning_rate=LEARNING_RATE,
	    optimizer="Adam",
	    momentum=MOMENTUM, # only needed for SGD
	    # MODEL
	    model_predefined=False,
	    checkpoint=None, # only needed if model_predefined=True
	    # DATASET PARAMS AND HYPERPARAMS
	    # for 16 GB without checking test_loss it's best to use 64 (if not logging test loss)
	    batch_size=32,
	    train_years=[2007, 2012],
	    batch_norm=True,
	    include_difficult=False,
	    transforms=True,
	    is_one_batch=False,
	    # OTHER PARAMS TO SET
	    fc_dropout=0.2, # for now this needs to be set manually in architecture.py!
	    grid_size=7, # in paper this is S
	    bounding_boxes=2, # in paper this is B
	    classes=20 # in paper this is C
	)

	# not passing just config, because wandb config is used with . instead of [""]
	train_dl_predef, test_dl_predef = dataset.prepare_data(config["batch_size"], config["include_difficult"], config["transforms"], config["train_years"])

	model, optimizer = train.model_pipeline(config, True, train_dl_predef, test_dl_predef)

	# to be added as param to execution
	save = False
	if save:
		utils.save_checkpoint(model, optimizer, "./yolo_test.pth.tar")

	# EVALUATION

