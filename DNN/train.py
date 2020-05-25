from dataset import PicDataset
import torch
import engine
from model import DNN
import config
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random

random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run():
    df = pd.read_csv(config.TRAIN_PATH)
    kfold = KFold(n_splits=5, random_state=config.SEED, shuffle=True)
    fold_losses = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print("-------------------------------------------------------")
        print(f"Training fold {i}")
        print("-------------------------------------------------------")
        train = df.iloc[train_idx]
        validation = df.iloc[val_idx]
        train_dataset = PicDataset(train)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE
        )

        val_dataset = PicDataset(validation)
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE
        )

        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        model = DNN()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        loss = 0

        for _ in range(config.EPOCHS):
            engine.train_fn(train_data_loader, model, optimizer, device)
            loss = engine.eval_fn(val_data_loader, model, device)
        print(f"Loss on fold {i} is {loss}")
        fold_losses.append(loss)
        torch.save(model.state_dict(), f'./models/model_{i}.bin')

    print(f"Average loss on cross validation is {sum(fold_losses) / 5}")


if __name__ == "__main__":
    run()

