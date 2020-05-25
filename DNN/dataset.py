import torch.nn as nn
import torch
import numpy as np


class PicDataset(nn.Module):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx].values.reshape(32, 32)
        target = x[8:24, 8:24].copy()
        x_top = x[:8, :].flatten()
        x_bottom = x[24:, :].flatten()
        x_left = x[8:24, :8].flatten()
        x_right = x[8:24, 24:].flatten()
        x = np.concatenate([x_top, x_bottom, x_left, x_right])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).view(-1)
