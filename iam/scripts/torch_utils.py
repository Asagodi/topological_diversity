import torch
import torch.nn as nn
import copy

class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, loss: float):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class BestModelSaver:
    def __init__(self, homeo_net: nn.Module, source_system: nn.Module):
        self.homeo_net = homeo_net
        self.source_system = source_system
        self.best_homeo_state = copy.deepcopy(homeo_net.state_dict())
        self.best_source_state = copy.deepcopy(source_system.state_dict())
        self.best_loss = float("inf")

    def step(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_homeo_state = copy.deepcopy(self.homeo_net.state_dict())
            self.best_source_state = copy.deepcopy(self.source_system.state_dict())

    def restore(self):
        self.homeo_net.load_state_dict(self.best_homeo_state)
        self.source_system.load_state_dict(self.best_source_state)
