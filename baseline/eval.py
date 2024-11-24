import torch
import torch.nn as nn
import torchvision.datasets as datasets
from baseline.data_transforms import crop10


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
        Args:
            X (torch.Tensor): batch_size x 3 x 224 x 224
        Returns:
            torch.Tensor: softmax output (m x num_classes)
        """
    model.eval()
    with torch.no_grad():
        X = crop10(X)
        res: torch.Tensor = model(next(X))
        for crop in X:
            res += model(crop)
        return res/10


def accuracy(model: nn.Module, dataset: datasets.VisionDataset,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    cv_loader = torch.utils.data.DataLoader(dataset, 512, num_workers=3)
    trues = 0
    for images, labels in cv_loader:
        images = images.to(device)
        labels = labels.to(device)
        predicted_labels = predict(model, images).argmax(1)
        trues += torch.eq(labels, predicted_labels).count_nonzero().item()
    return float(trues) / len(dataset)


def topk(model: nn.Module, dataset: datasets.VisionDataset, k=1,
         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    cv_loader = torch.utils.data.DataLoader(dataset, 512, num_workers=3)
    trues = 0
    for images, labels in cv_loader:
        images = images.to(device)
        labels = labels.to(device)
        predicted_labels, _ = predict(model, images).topk(1)
        trues += predicted_labels.eq_(
            labels.reshape(-1, 1)).count_nonzero().item()
    return float(trues) / len(dataset)
