import torch
import torch.nn as nn
from torchvision import datasets


def pca(
    dataset: datasets.VisionDataset,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform PCA on whole dataset

    Returns:
        eigenvalues, eigenvectors
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, num_workers=3)
    # calculate covariance matrix of RGB over dataset
    covar = torch.zeros(3, 3, device=device)
    for images, _ in loader:
        images: torch.Tensor = images.to(device)
        # permute RGB channels to dim 0
        images = images.permute(1, 0, 2, 3).flatten(start_dim=1)
        # dataset D are already centered around zero
        # D: 3 x (batch_size * h * w)
        # covar = DD^T / N = Sum(XX^T)/N (partitioned matrices)
        covar.addmm_(images, images.T)
    # calculate eigenvalues and eigenvectors
    covar /= len(dataset)*256*256
    print(f'Covar:\n{covar}')
    return torch.linalg.eigh(covar)


class PCAColorAugmentation(nn.Module):
    def __init__(self, eigvals, eigvecs, *args, **kwargs):
        super(PCAColorAugmentation, self).__init__(*args, **kwargs)
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.eigvals: torch.Tensor = eigvals.sqrt().reshape(3, 1).cpu()
        self.eigvecs: torch.Tensor = eigvecs.cpu()
        print(f'Eigen values (sqrt):\n{self.eigvals}')
        print(f'Eigen vectors:\n{self.eigvecs}')

    def forward(self, X: torch.Tensor):
        # draw alpha once for each image
        alpha = torch.normal(0, 0.1, size=(3, 1))
        noise: torch.Tensor = self.eigvecs.mm(
            alpha.mul_(self.eigvals))
        X += noise.reshape(3, 1, 1)
        return X
