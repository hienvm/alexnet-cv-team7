import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from sklearn.datasets import load_sample_image
import numpy as np
from itertools import chain

preprocess = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToImage(),
    # scale to [0, 1)
    v2.ToDtype(torch.float32, scale=True),
    # ImageNet Mean & StdDeviation
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# training data augmentation
train_augment = v2.Compose([
    v2.RandomCrop(224),
    v2.RandomHorizontalFlip(),
    # TODO: PCA color augmentation
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
    return chain(crop5(X), crop5(X.fliplr()))


if __name__ == '__main__':
    img = tv_tensors.Image(np.permute_dims(
        load_sample_image('flower.jpg').copy(), (2, 0, 1)), dtype=torch.uint8).unsqueeze(0)
    a = crop10(preprocess(img))
    print(next(a).shape)
    for i, x in enumerate(a):
        print(f'{i}.')
