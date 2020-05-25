import torch.nn as nn
import torch


class PicDataset(nn.Module):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx].values.reshape(32, 32).copy()
        target = x[8:24, 8:24].copy()
        x[8:24, 8:24] = 1

        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(target, dtype=torch.float32).unsqueeze(0)