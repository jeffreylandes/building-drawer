import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from training.data import BuildingData
from training.utils.train_utils import create_network, update_network, get_reward

NUM_DATA_WORKERS = 3


def train(parameters):
    torch.set_default_dtype(torch.float32)

    data = BuildingData()
    dataloader = DataLoader(
        data, batch_size=parameters.batch_size, shuffle=True, num_workers=NUM_DATA_WORKERS
    )

    criterion = MSELoss()

    actor_direction = create_network("actor_direction", parameters.learning_rate)
    critic_direction = create_network("critic_direction", parameters.learning_rate)
    actor_distance = create_network("actor_distance", parameters.learning_rate)
    critic_distance = create_network("critic_distance", parameters.learning_rate)

    for epoch in range(parameters.num_epochs):
        for step, data_item in enumerate(dataloader):
            site = data_item["site"]
            distance = data_item["distance"]
            direction = data_item["direction"]
            mask = data_item["target_mask"]

            # Action
            prediction_action_direction = actor_direction.model(site, mask)
            prediction_action_distance = actor_distance.model(site, direction)

            prediction_critic_direction = critic_direction.model(site, prediction_action_direction)
            prediction_critic_distance = critic_direction.model(site, direction, prediction_action_distance)

            reward_direction = get_reward(prediction_action_direction, direction)
            reward_distance = get_reward(prediction_action_distance, distance)

            loss_critic_direction = criterion(reward_direction, prediction_critic_direction)
            loss_critic_distance = criterion(reward_distance, prediction_critic_distance)

            update_network(critic_direction, loss_critic_direction)
            update_network(critic_distance, loss_critic_distance)

            # TODO: Update actor according to sampled policy gradient


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=35)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    opt = parser.parse_args()

    train(opt)
