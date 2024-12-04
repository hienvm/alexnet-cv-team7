import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.datasets as datasets
from improved.eval import accuracy
# from tqdm.notebook import trange
from time import time
from typing import Literal

def train(model: nn.Module,
          train_dataset: datasets.VisionDataset,
          cv_dataset: datasets.VisionDataset,  # cross validation
          batch_size=128,
          num_epochs=90,
          class_weights: torch.Tensor | None = None,
          num_workers=2,
          initial_lr=0.001,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          patience=10,
          optimizer: Literal['adam'] | Literal['sgd']='sgd',
          output_type: Literal['softmax'] | Literal['sigmoid']='softmax',
          ):
    train_start_time = time()
    epoch_costs = []
    cv_error_rates = []
    learning_rates = []

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=initial_lr, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    # reduce lr 10 times whenever validation error doesn't reduce after patience epochs
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=patience)
    
    cost_func = nn.CrossEntropyLoss(weight=class_weights) if output_type=='softmax' \
            else nn.BCEWithLogitsLoss(weight=class_weights)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # epoch 0
    # error_rate = 1 - accuracy(model, cv_dataset, device)
    # cv_error_rates.append(error_rate)
    # learning_rates.append(lr_scheduler.get_last_lr())

    for i in range(num_epochs):
        epoch_start_time = time()
        model.train()
        cost_sum = 0.0
        for images, labels in train_loader:
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            
            output: torch.Tensor = model(images)
            
            if output_type == 'sigmoid':
                labels = torch.zeros(output.shape) \
                    .scatter_(1, labels.reshape(labels.shape[0], 1), 1)
                
            cost: torch.Tensor = cost_func(output, labels)

            del images
            del labels
            del output
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            cost_sum += cost.item()

        # epoch cost
        avg_cost = cost_sum / len(train_loader)
        epoch_costs.append(avg_cost)
        # validation error rate
        error_rate = 1 - accuracy(model, cv_dataset)
        cv_error_rates.append(error_rate)
        # reduce lr 10 times whenever validation error doesn't reduce after num_epochs/10 epochs
        lr_scheduler.step(error_rate)
        learning_rates.append(lr_scheduler.get_last_lr()[0])
        
        print(f'Epoch {i+1}/{num_epochs}, Cost: {avg_cost:.3f}, Val Error: {error_rate:.2%}, lr: {lr_scheduler.get_last_lr()[0]}, Time: {time()-epoch_start_time:.0f}s')

    # last epoch
    # with torch.no_grad():
    #     cost_sum = 0.0
    #     for images, labels in train_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         cost: torch.Tensor = cross_entropy(model(images), labels)
    #         cost_sum += cost.item()
    #     epoch_costs.append(cost_sum / len(train_loader))
        
    print(f'Training time: {(time()-train_start_time)/60:.0f}m')

    return epoch_costs, cv_error_rates, learning_rates
