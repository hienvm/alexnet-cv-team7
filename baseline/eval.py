import torch
import torch.nn as nn
import torchvision.datasets as datasets
from baseline.data_transforms import crop10


def predict(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
        Args:
            X (torch.Tensor): m(kích thước batch) x 3 x 224 x 224

        Returns:
            torch.Tensor: softmax output (m x num_classes)
        """
    model.eval()
    with torch.no_grad():
        X = crop10(X)
        res = model(next(X))
        for crop in X:
            res += model.__call__(crop)
        return res/10


def accuracy(model: nn.Module, dataset: datasets.VisionDataset,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    cv_loader = torch.utils.data.DataLoader(dataset, 64)
    num_true = 0
    for i, (images, labels) in enumerate(cv_loader):
        images = images.to(device)
        labels = labels.to(device)
        predicted_labels = predict(model, images).argmax(1)
        num_true += torch.eq(labels, predicted_labels).count_nonzero().item()
    return 1 - float(num_true) / len(dataset)
