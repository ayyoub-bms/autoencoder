import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim

from research.pytorch.earlystop import EarlyStopping
from research.config import plots

plots.init()
plots.toggle_grid(True)


def train_model(model, device, training_set, loss_function, optimizer,
                lasso_param):

    model.train()
    running_loss = batch = 0

    for batch, (char, port, rets) in enumerate(training_set):
        char = char.to(device)
        port = port.to(device)
        rets = rets.to(device)

        optimizer.zero_grad()

        # forward step
        predictions = model(char, port)

        # Loss computation
        current_loss = loss_function(predictions, rets)
        current_loss += model.lasso(lasso_param)
        running_loss += current_loss.item()

        # Back propagation
        current_loss.backward()

        # Param update
        optimizer.step()

    return running_loss / (batch + 1)


def validate_model(model, device, validation_set, loss_function):

    running_loss = 0

    model.eval()
    with torch.no_grad():
        for batch, (char, port, rets) in enumerate(validation_set):
            char = char.to(device)
            port = port.to(device)
            rets = rets.to(device)

            predictions = model(char, port)
            loss = loss_function(predictions, rets)
            running_loss += loss.item()

    return running_loss / (batch + 1)


def run(model, device, training_set, validation_set,
        epochs, lasso_param, learning_rate,
        save_plots=False, fig=None, fig_title=None):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    early_stopping = EarlyStopping(max_iter=20)

    valid_losses = np.full(epochs, np.nan)
    train_losses = np.full(epochs, np.nan)
    stop = 0

    for t in range(epochs):
        train_loss = train_model(model,
                                 device,
                                 training_set,
                                 loss_function,
                                 optimizer,
                                 lasso_param)

        valid_loss = validate_model(model,
                                    device,
                                    validation_set,
                                    loss_function)

        train_losses[t] = train_loss
        valid_losses[t] = valid_loss

        early_stopping(model, valid_loss)

        if early_stopping.stop:
            stop = t - early_stopping.max_iter
            model = early_stopping.best_model
            validation_score = early_stopping.best_score
            break
        else:
            validation_score = valid_loss

    if save_plots:
        assert fig is not None and fig_title is not None
        fig.clf()
        ax = fig.gca()
        ax.plot(train_losses, label='Training', color='black')
        ax1 = plt.twinx()
        ax1.plot(valid_losses, label='Validation', color='red')
        if stop > 0:
            ax1.axvline(stop, label='Early stopping', ls='--', color='blue')
        fig.legend()
        fig.suptitle(fig_title)
        fig.savefig(fr'outputs/{fig_title}.pdf')

    return validation_score, model
