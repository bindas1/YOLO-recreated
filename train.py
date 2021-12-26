from loss import YoloLoss
import torch
import wandb
import architecture
import dataset
import utils
import evaluation
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_pipeline(hyp, data_predefined=False, train_dl=None, test_dl=None, device=device):
    with wandb.init(project="YOLO-recreated", entity="bindas_try", config=hyp):
        config = wandb.config
        
        # make the model, data, and optimization problem
        model, train_dl, test_dl, criterion, optimizer = make(config, data_predefined, train_dl, test_dl)
        
        # and use them to train the model
        train(model, train_dl, test_dl, criterion, optimizer, config)

        # can be moved to evaluation func
        # utils.save_checkpoint(model, optimizer, "../input/utility-for-yolo/test_check.pth.tar")

        # and test its final performance
        evaluation.evaluate_model(model, test_dl, config, test_dl=True)
        
    return model, optimizer

def make(config, data_predefined, train_dl_predef, test_dl_predef):
    if data_predefined:
        train_dl, test_dl = train_dl_predef, test_dl_predef
    else:
        train_dl, test_dl = dataset.prepare_data(
            config.batch_size, config.include_difficult, config.transforms, config.train_years
        )
        
    if config.is_one_batch:
        train_dl = next(iter(train_dl))
        test_dl = next(iter(test_dl))

    # Make the model
    model = architecture.darknet(config.batch_norm)
    model.to(device)
    
    # Make the loss and optimizer
    criterion = YoloLoss()

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate
        )
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=config.momentum,
            lr=config.learning_rate
        )
    else:
        print("this model is not supported, please choose ADAM or SGD")

    # load params to model if wanted
    if config.model_predefined:
        utils.load_checkpoint(config.checkpoint, model, optimizer)
        model.train()
    
    return model, train_dl, test_dl, criterion, optimizer
        

def train(model, train_dl, test_dl, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all")
    
    # enumerate epochs
    for epoch in tqdm(range(config.epochs)):
        running_loss = 0.0
        running_val_loss = 0.0
        idx = 1
        
        if not config.is_one_batch:
            for i, (inputs, _, targets) in enumerate(train_dl):
                loss, batch_size = train_batch(inputs, targets, model, optimizer, criterion)
                running_loss += loss.item() * batch_size
                if i%100==0:
                    print(loss.item(), epoch)
            running_loss = running_loss / len(train_dl)

            for i, (inputs, _, targets) in enumerate(test_dl):
                val_loss = test_batch(inputs, targets, model, criterion)
                if i%100==0:
                    print("VAL LOSS {}, {}".format(val_loss, epoch))
                running_val_loss += val_loss * batch_size
                # if idx % 25 == 0:
                #     wandb.log({
                #         "test_loss_batches": running_val_loss / idx
                #     }, step=idx//25)
                # idx += 1
            running_val_loss = running_val_loss / len(test_dl)
        else:
            with torch.autograd.detect_anomaly():
                # for one batch only
                loss, batch_size = train_batch(train_dl[0], train_dl[2], model, optimizer, criterion)
                running_loss = loss.item() * batch_size

                val_loss = test_batch(test_dl[0], test_dl[2], model, criterion)
                running_val_loss = val_loss * batch_size

        wandb.log({
            "epoch": epoch, 
            "avg_batch_loss": running_loss,
            "test_loss": running_val_loss
        }, step=epoch)
#         wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print("Average epoch loss {}".format(running_loss))


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    
    size = images.size(0)
    del images, labels

    return loss, size
    # return loss, images.size(0)

def test_batch(val_images, val_labels, model, criterion):
    # get test loss
    model.eval()

    val_images, val_labels = val_images.to(device), val_labels.to(device)

    with torch.no_grad():
        val_predictions = model(val_images)
    test_loss = criterion(val_predictions, val_labels)

    model.train()
    return test_loss.item()

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")