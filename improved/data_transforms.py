import torch
from torchvision.transforms import v2
from torchvision import datasets
from itertools import chain
from baseline.pca import PCAColorAugmentation, pca

# used to calculate mean and get final preprocess
prepreprocess = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToImage(),
    v2.ToDtype(torch.float32),
])


def calc_mean(
    dataset: datasets.VisionDataset,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    with torch.no_grad():
        mean = torch.zeros(3, device=device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, num_workers=3)
        for images, _ in loader:
            images = images.to(device)
            mean += images.flatten(start_dim=2).mean(dim=2).sum(dim=0)
        return mean.div_(len(dataset)).tolist()


def get_preprocess(dataset: datasets.VisionDataset, mean, std):
    mean = calc_mean(dataset)
    print(f'Mean: {mean}')
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        # Only subtracts mean
        v2.Normalize(mean, (1, 1, 1)),
    ])


def get_train_augment(dataset: datasets.VisionDataset):
    # training data augmentation
    eigvals, eigvecs = pca(dataset)
    return v2.Compose([
        prepreprocess,
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(0.5),
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
    return chain(crop5(X), crop5(X.fliplr()))


# if __name__ == '__main__':
    # print(calc_mean(datasets.CIFAR10(
    #     'datasets/cifar10', train=True, transform=lambda X: train_augment(prepreprocess(X)))))

    # a = train_augment(prepreprocess(img))
    # print(a.shape)
    #     for i, x in enumerate(a):
    #         print(f'{i}.')
