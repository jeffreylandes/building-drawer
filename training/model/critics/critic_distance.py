import torch
from torch.nn import Module, Sequential, ReLU, MaxPool2d, Linear

from training.model.layers import CNNBlock


class CriticDistance(Module):
    def __init__(self):
        super(CriticDistance, self).__init__()

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
            Linear(292, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 8),
            ReLU(),
            Linear(8, 1),
        )

    def forward(self, state, direction, distribution_distance):
        feature_map = self.cnn_layers(state)
        feature_map_squeezed = feature_map.reshape((state.shape[0], -1))
        feature_map_actions = torch.cat(
            [feature_map_squeezed, direction, distribution_distance], dim=1
        )
        feature_map_final = self.linear_layers(feature_map_actions)
        return feature_map_final
