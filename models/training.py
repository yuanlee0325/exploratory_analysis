import torch
from torch import nn



ce = nn.CrossEntropyLoss()
def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc

def train_batch(model, data, optimizer, criterion, device):
    model.train()
    ims, ce_masks, _ = data
    ims = ims.to(device)
    ce_masks = ce_masks.to(device)
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion, device):
    model.eval()
    ims, masks, _ = data
    ims = ims.to(device)
    masks = masks.to(device)
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()