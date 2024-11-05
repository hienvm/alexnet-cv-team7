import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            output: torch.Tensor = self.__call__(X)
            return output.argmax(1).numpy()
