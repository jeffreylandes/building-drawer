from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Normal
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
        return Network(model=model, optimizer=optimizer)
    elif network_type == "actor_distance":
        model = ActorDistance()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(model=model, optimizer=optimizer)
    elif network_type == "critic_distance":
        model = CriticDistance()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(model=model, optimizer=optimizer)
    elif network_type == "critic_direction":
        model = CriticDirection()
        model.train()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)
        return Network(model=model, optimizer=optimizer)
    else:
        raise Exception(f"Unsupported model: {network_type}")


def copy_model(model: Module):
    return deepcopy(model)


def update_network(network: Network, loss: torch.tensor):
    network.optimizer.zero_grad()
    loss.backward()
    network.optimizer.step()


def get_reward(distributions: torch.tensor, targets: torch.tensor):
    losses = []
    for i, (mean, std, _) in enumerate(distributions):
        # TODO: Would be nice to get rid of this somehow...
        std = std ** 2
        std = torch.tensor(1e-9) if std.item() == 0 else std
        m = Normal(torch.tensor([mean]), torch.tensor([std]))
        target = targets[i]
        loss = 0
        for _ in range(NUM_SAMPLES):
            sample = m.sample()
            loss += np.abs(sample - target) ** 2
        loss /= NUM_SAMPLES
        losses.append(loss)
    rewards = torch.tensor(-np.array(losses).astype(np.float32))
    return rewards
