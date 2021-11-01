import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from training.data import BuildingData
from training.model.cnn import CNN


NUM_DATA_WORKERS = 3


def train(parameters):
    torch.set_default_dtype(torch.float32)
    data = BuildingData()
    dataloader = DataLoader(
        data, batch_size=opt.batch_size, shuffle=True, num_workers=NUM_DATA_WORKERS
    )

    criterion = L1Loss()

    model = CNN()
    optimizer = Adam(params=model.parameters(), lr=0.0001)

    model.train()

    for epoch in range(parameters.num_epochs):
        epoch_loss = 0
        for step, data_item in enumerate(dataloader):
            site = data_item["site"]
            target = data_item["target"]
            mask = data_item["target_mask"]

            prediction = model(site, mask)
            loss = criterion(prediction, target)

            if step % parameters.log_interval == 0:
                print(f"Step loss: {loss.item()}")
                print(prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch loss: {epoch_loss}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    opt = parser.parse_args()

    train(opt)
