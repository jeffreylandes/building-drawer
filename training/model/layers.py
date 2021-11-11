from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU


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
