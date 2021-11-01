from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear


class CNNBlock(Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, activation=ReLU()
    ):
        super(CNNBlock, self).__init__()

        self.layer = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.layer(x)


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            CNNBlock(in_channels=3, out_channels=5, kernel_size=7, stride=1, padding=1),
            CNNBlock(in_channels=5, out_channels=8, kernel_size=5, stride=1, padding=1),
            MaxPool2d(2, 2),
            CNNBlock(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1),
            MaxPool2d(2, 2),
            CNNBlock(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            MaxPool2d(2, 2),
            CNNBlock(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            MaxPool2d(2, 2),
            CNNBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            MaxPool2d(2, 2),
            CNNBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            MaxPool2d(2, 2),
            CNNBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            MaxPool2d(2, 2),
        )

        self.linear_layers = Sequential(
            Linear(1152, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 8),
            ReLU(),
            Linear(8, 3),
        )

    def forward(self, x, mask):
        feature_map = self.cnn_layers(x)
        feature_map_squeezed = feature_map.reshape((x.shape[0], -1))
        feature_map_final = self.linear_layers(feature_map_squeezed)
        return feature_map_final * mask
