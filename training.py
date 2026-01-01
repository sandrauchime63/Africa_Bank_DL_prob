from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm

def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    training_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)       
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

        probs = torch.sigmoid(output)
        preds = (probs > 0.5).long()

        correct += (preds == targets.long()).sum().item()
        total += targets.size(0)

    train_acc = correct / total
    avg_loss = training_loss / total

    return avg_loss, train_acc



def predict(model, loss_fn, data_loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct=0
    total=0

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)   # SCALAR loss
            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            correct += (preds == targets.long()).sum().item()
            total += targets.size(0)
    val_acc = correct / total
    avg_loss = total_loss / total
    
    return avg_loss, val_acc

