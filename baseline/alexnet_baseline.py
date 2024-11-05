import torch
import torch.nn as nn
from classfier import Classifier


class AlexNet(Classifier):
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 1 device
        self.conv_layers = [
            nn.Sequential(
                # conv_layer 1
                nn.Conv2d(in_channels=3, out_channels=96,
                          kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                # conv_layer 2
                nn.Conv2d(in_channels=96, out_channels=256,
                          kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                # conv_layer 3
                nn.Conv2d(in_channels=256, out_channels=384,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # conv_layer 4
                nn.Conv2d(in_channels=384, out_channels=384,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # conv_layer 5
                nn.Conv2d(in_channels=384, out_channels=256,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
        ]

        self.fc_layers = [
            nn.Sequential(
                # fc_layer 1
                nn.Dropout(0.5),
                nn.Linear(in_features=9216, out_features=4096),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # fc_layer 2
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                # fc_layer 3
                nn.Linear(in_features=4096, out_features=num_classes),
            ),
        ]

        self.network = nn.Sequential(
            *self.conv_layers,
            nn.Flatten(),
            *self.fc_layers,
            nn.Softmax(1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # input dimensions: m(batch size) x 3 x 224 x 224
        return self.network(X)


if __name__ == "__main__":
    # test
    input = nn.Parameter(torch.rand(2, 3, 224, 224))
    model = AlexNet(10)
    output: torch.Tensor = model(input)
    output.sum().backward()
    print(model(input))
    print(model.predict(input))
