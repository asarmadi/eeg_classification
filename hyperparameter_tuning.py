import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from utils.config import Config
from utils.utils import *

from ray import train, tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser(description='EEG Classification')
parser.add_argument('--model_type', default='cnn1d',type=str, help='cnn, caspnet, lstm')
parser.add_argument('--batch_size', default=32, type=int, help='Test Batch Size')
parser.add_argument('--n_epochs', default=500, type=int, help='Number of epochs')
args = parser.parse_args()

configGeneral = Config(args.model_type)

def train_func(model, optimizer, train_loader):
    model.train()
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device).reshape(-1,1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test_func(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target,_) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.round()
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

def train_eeg(config):
    # Data Setup
    train_loader, test_loader, _ = data_loader(64, 4, args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_loader(configGeneral, config["kern"])
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    for i in range(10):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)

        # Send the current training result back to Tune
        train.report({"mean_accuracy": acc})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")

config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        "kern": tune.sample_from(lambda _: (2 * np.random.randint(2, 20)-1)),
    }


tuner = tune.Tuner(
    tune.with_resources(train_eeg, {"gpu": 1}),
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    ),
    param_space=config,
)
results = tuner.fit()

print(results.get_best_result("mean_accuracy", mode="max"))

# To enable GPUs, use this instead:
# analysis = tune.run(
#     train_mnist, config=search_space, resources_per_trial={'gpu': 1})
#import os

#logdir = results.get_best_result("mean_accuracy", mode="max").path
#state_dict = torch.load(os.path.join(logdir, "model.pth"))

#model = ConvNet()
#model.load_state_dict(state_dict)
