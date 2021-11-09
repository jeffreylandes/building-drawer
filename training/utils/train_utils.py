from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn import Module
from torch.optim import Adam

from training.model.critics.critic_direction import CriticDirection
from training.model.critics.critic_distance import CriticDistance
from training.model.actors.actor_distance import ActorDistance
from training.model.actors.actor_direction import ActorDirection


NUM_SAMPLES = 10


@dataclass
class Network:
    model: Module
    optimizer: Adam


def create_network(network_type: str, learning_rate: float):
    if network_type == "actor_direction":
        model = ActorDirection()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(
            model=model,
            optimizer=optimizer
        )
    elif network_type == "actor_distance":
        model = ActorDistance()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(
            model=model,
            optimizer=optimizer
        )
    elif network_type == "critic_distance":
        model = CriticDistance()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(
            model=model,
            optimizer=optimizer
        )
    elif network_type == "critic_direction":
        model = CriticDirection()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(
            model=model,
            optimizer=optimizer
        )
    else:
        raise Exception(f"Unsupported model: {network_type}")


def copy_model(model: Module):
    return deepcopy(model)


def update_network(network: Network, loss: torch.tensor):
    network.optimizer.zero_grad()
    loss.backward()
    network.optimizer.step()


def get_reward(distribution: Tuple[float, float], target: float):
    for i in range(NUM_SAMPLES):
        # TODO: sample from distribution
        # TODO: get distance from target
        # TODO: average
        pass
