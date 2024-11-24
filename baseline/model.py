import torch
import torch.nn as nn


def init_params(layer: nn.Module):
    '''Initialize parameters'''
    if not isinstance(layer, nn.Conv2d) and not isinstance(layer, nn.Linear):
        return
    if not isinstance(layer, nn.Conv2d) or layer.in_channels not in (3, 256):
        # set all biases to 1 except in 1st and 3rd conv layer
        torch.nn.init.ones_(layer.bias)
    # set all weights to ~ N(0, 0.01)
    torch.nn.init.normal_(layer.weight, 0, 0.01)


class AlexNetBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        *args, **kwargs
    ):
        """
        Args:
            num_classes (int): số lượng lớp
        """
        super(AlexNetBaseline, self).__init__(*args, **kwargs)

        # 1 device
        self.conv_layers = [
            nn.Sequential(
                # conv_layer 1
                nn.Conv2d(in_channels=3, out_channels=96,
                          kernel_size=11, stride=4, padding=2, device=device),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                # conv_layer 2
                nn.Conv2d(in_channels=96, out_channels=256,
                          kernel_size=5, stride=1, padding=2, device=device),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                # conv_layer 3
                nn.Conv2d(in_channels=256, out_channels=384,
                          kernel_size=3, stride=1, padding=1, device=device),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # conv_layer 4
                nn.Conv2d(in_channels=384, out_channels=384,
                          kernel_size=3, stride=1, padding=1, device=device),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # conv_layer 5
                nn.Conv2d(in_channels=384, out_channels=256,
                          kernel_size=3, stride=1, padding=1, device=device),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
        ]

        self.fc_layers = [
            nn.Sequential(
                # fc_layer 1
                nn.Dropout(0.5),
                nn.Linear(in_features=9216, out_features=4096, device=device),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # fc_layer 2
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=4096, device=device),
                nn.ReLU(inplace=True),
            ),
            # nn.Sequential(
            # fc_layer 3
            nn.Linear(in_features=4096,
                      out_features=num_classes, device=device),
            # ),
        ]

        self.network = nn.Sequential(
            *self.conv_layers,
            nn.Flatten(),
            *self.fc_layers,
            nn.Softmax(1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): m(kích thước batch) x 3 x 224 x 224

        Returns:
            torch.Tensor: softmax output (m x num_classes)
        """
        return self.network(X)


if __name__ == "__main__":
    input = torch.rand(2, 3, 224, 224)
    model = AlexNetBaseline(5)
    output: torch.Tensor = model(input)
    output.sum().backward()
    print(model(input).shape)
