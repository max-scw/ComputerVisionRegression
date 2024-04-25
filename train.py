from pathlib import Path
from timeit import default_timer
import sys

from tqdm import tqdm
import numpy as np
import copy

import torch
import torch.nn as nn


from utils.dataset import LoadImagesAndLabels, build_dataloaders
from model import build_model


def train(
        model,
        dataloaders: dict,
        num_epochs: int,
        batch_size: int = 4,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device: str = "cpu"
):
    if criterion is None:
        criterion = nn.MSELoss()  # mean square error
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if scheduler is None:
        scheduler = optimizer

    t0 = default_timer()

    # initialize best_loss and model_weights
    best_loss = np.inf
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    # progress bar object
    pbar = tqdm(range(num_epochs), ascii=True, desc="Epoch ", file=sys.stdout)
    for epoch in pbar:
        # update progress bar
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        desc = {}
        # Each epoch has a training and validation phase
        for phase in ["training", "validation"]:
            is_training = phase == "training"

            if is_training:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_training:
                        loss.backward(retain_graph=False)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if is_training:
                scheduler.step()

            # calculate loss
            epoch_loss = running_loss / len(dataloaders[phase])
            # store information to update the bar
            desc[f"{phase}_loss"] = epoch_loss

            # deep copy the model
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        # update postfix of the progress bar
        pbar.set_postfix(desc)
        history.append(desc)

    time_elapsed = default_timer() - t0
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, history


if __name__ == "__main__":
    info_files = {
        "training": Path(r"./Trn.txt"),
        "validation": Path(r"./Val.txt"),
    }

    # get model
    mdl, transforms = build_model()
    batch_size = 4

    data_loaders = build_dataloaders(info_files, transforms, batch_size=batch_size)

    mdl, hist = train(mdl, data_loaders, num_epochs=100, batch_size=batch_size)
