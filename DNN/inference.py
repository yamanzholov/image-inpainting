from dataset import PicDataset
from model import DNN
import config
import torch
import pandas as pd
import numpy as np


def predict():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(config.TEST_PATH, header=None)
    dataset = PicDataset(df.loc[:, 1:])
    preds = np.zeros((len(dataset), 256))

    for i in range(5):
        temp = np.zeros((len(dataset), 256))
        model = DNN()
        model.load_state_dict(torch.load(f'./models/model_{i}.bin'))
        model.to(device)
        model.eval()
        for j in range(len(dataset)):
            x, _ = dataset[j]
            x = x.to(device)
            y = model(x)
            temp[j, :] = y.detach().cpu().numpy()
        preds += temp

    preds /= 5
    df = pd.DataFrame(np.concatenate([np.arange(1, 921).reshape(-1, 1), preds], axis=1), columns=np.arange(257))
    df[0] = df[0].astype('int')
    df.to_csv('./predictions.csv', index=False, header=False)


if __name__ == "__main__":
    predict()
