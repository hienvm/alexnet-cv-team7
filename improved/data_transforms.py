import torch
from torchvision.transforms import v2
from torchvision import datasets
from itertools import chain
from improved.pca import PCAColorAugmentation, pca

# used to calculate mean and get final preprocess
prepreprocess = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


def calc_mean_std(
    dataset: datasets.VisionDataset,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    with torch.no_grad():
        mean = torch.zeros(3, device=device)
        var = torch.zeros(3, device=device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, num_workers=3)
        size = len(dataset)
        # mean
        for images, _ in loader:
            images = images.to(device)
            mean += images.flatten(start_dim=2).mean(dim=2).sum(dim=0)
        mean = mean.div_(size).reshape(1,3,1)
        # variance
        for images, _ in loader:
            images = images.to(device)
            var += images.flatten(start_dim=2).sub_(mean).square_().sum(2).sum(0)
        return mean.reshape(3).tolist(), var.div_(size*256*256).sqrt_().tolist()


def get_preprocess(mean, std):
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


def get_train_augment(eigvals, eigvecs, mean, std):
    # training data augmentation
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(0.5),
        v2.TrivialAugmentWide(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
        PCAColorAugmentation(eigvals, eigvecs),
    ])



def crop(X: torch.Tensor, heightOffset: int, widthOffset: int):
    return X[..., heightOffset:224+heightOffset, widthOffset:224+widthOffset]


def crop5(X: torch.Tensor):
    # top left
    yield crop(X, 0, 0)
    # bottom left
    yield crop(X, 32, 0)
    # top right
    yield crop(X, 0, 32)
    # bottom right
    yield crop(X, 32, 32)
    # center
    yield crop(X, 16, 16)


def crop10(X: torch.Tensor):
    """Use generator to save space.

    Yields:
        10 crops for each image in batch
    """
    return chain(crop5(X), crop5(v2.functional.hflip(X)))