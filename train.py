def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_num = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu()
        train_num += data.shape[0]
    return train_loss / train_num
