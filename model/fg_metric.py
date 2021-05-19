import torch
import torch.nn as nn
import numpy as np
import cupy as cp


def foreground_loss(y_pred, y_true):
    criterion = nn.BCELoss()
    loss = criterion(y_pred, y_true)
    return loss

def foreground_accuracy(y_pred, y_true):
    round_pred = torch.round(y_pred)
    return torch.mean(torch.eq(round_pred, y_true).float())
    