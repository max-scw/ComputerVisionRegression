from pathlib import Path
from timeit import default_timer
import sys

from tqdm import tqdm
import numpy as np
import copy
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils.dataset import LoadImagesAndLabels, build_dataloader
from model import build_model

from typing import List


def train(
        model,
        dataloader_training: DataLoader,
        num_epochs: int,
        dataloader_validation: DataLoader,
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

    dataloaders = {
        "training": dataloader_training,
        "validation": dataloader_validation if isinstance(dataloader_validation, DataLoader) else dataloader_training,
    }

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


def predict(
        model,
        dataloader: DataLoader,
        device: str = "cpu"
) -> List[float]:
    model.eval()

    # Iterate over data.
    output = []
    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        output.append(outputs)
    return list(chain.from_iterable(output))


if __name__ == "__main__":

    # get model
    mdl, transforms = build_model()
    batch_size = 21

    data_loader_train = build_dataloader(Path(r"./Trn.txt"), transforms, batch_size=batch_size, shuffle_data=True)
    data_loader_valid = build_dataloader(Path(r"./Val.txt"), transforms, batch_size=batch_size)
    mdl, hist = train(
        mdl,
        data_loader_train,
        dataloader_validation=data_loader_valid,
        num_epochs=60
    )

    # df = pd.DataFrame(hist)
    # df.plot()
    # plt.show()


    data_loader_test = build_dataloader(Path(r"./Tst.txt"), transforms)
    results = predict(mdl, data_loader_test)
