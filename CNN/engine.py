import torch.nn as nn
from tqdm.autonotebook import tqdm
import utils
import config
import torch


def train_fn(data_loader, model, optimizer, device):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    losses = utils.AverageMeter()
    criterion = nn.MSELoss()

    for i, data in enumerate(tk0):
        x, target = data
        x = x.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(target, preds)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), config.BATCH_SIZE)
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for i, data in enumerate(tk0):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            criterion = nn.MSELoss()
            preds = model(x)
            loss = criterion(preds, target)
            losses.update(loss.item(), config.BATCH_SIZE)
            tk0.set_postfix(loss=losses.avg)
    print(f"Validation loss: {losses.avg}")

    return losses.avg
