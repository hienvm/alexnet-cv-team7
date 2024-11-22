import torch
import torch.nn as nn
import torchvision.datasets as datasets
from baseline.eval import accuracy
from tqdm import tqdm_notebook as tqdm


def train(model: nn.Module,
          train_dataset: datasets.VisionDataset,
          cv_dataset: datasets.VisionDataset,  # cross validation
          batch_size=128,
          num_epochs=90,
          class_weights: torch.Tensor | None = None,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          ):
    epoch_costs = []
    cv_error_rates = []
    learning_rates = []

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # reduce lr 10 times whenever validation error doesn't reduce after num_epochs/10 epochs
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=num_epochs/10, min_lr=1e-5)
    cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2)

    # epoch 0
    error_rate = accuracy(model, cv_dataset, device)
    cv_error_rates.append(error_rate)
    learning_rates.append(lr_scheduler.get_last_lr())

    for i in tqdm(range(num_epochs)):
        model.train()
        cost_sum = 0.0
        for j, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            cost: torch.Tensor = cross_entropy(model(images), labels)
            cost_sum += cost.item()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        # epoch cost
        epoch_costs.append(cost_sum / len(train_loader))
        # validation error rate
        error_rate = accuracy(model, cv_dataset)
        cv_error_rates.append(error_rate)
        # reduce lr 10 times whenever validation error doesn't reduce after num_epochs/10 epochs
        lr_scheduler.step(error_rate)
        learning_rates.append(lr_scheduler.get_last_lr())

    # last epoch
    with torch.no_grad():
        cost_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            cost: torch.Tensor = cross_entropy(model(images), labels)
            cost_sum += cost.item()
        epoch_costs.append(cost_sum / len(train_loader))

    return epoch_costs, cv_error_rates, learning_rates
