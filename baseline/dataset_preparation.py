from numpy import random


# def train_cv_split(dataset: datasets.VisionDataset, cv_ratio=0.1):
#     size = len(dataset)
#     return torch.utils.data.random_split(dataset, [size*(1-cv_ratio), size*cv_ratio])


def indices_split(size: int, ratio: float):
    '''randomly splits into {ratio*size, (1-ratio)*size}'''
    indices: list[int] = random.choice(
        size, int(ratio*size), replace=False).tolist()
    return indices, list(set(range(size)).difference(set(indices)))
