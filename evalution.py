import sklearn.metrics as metrics
import torch


def evaluate(model, val_loader, criterion, scheduler, device):
    model.eval()
    cur_label = []
    cur_pred = []
    val_loss = 0
    valid_acc = 0
    val_num = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = criterion(pred, target)

            cur_label.extend(target.cpu().tolist())
            pred_result = torch.argmax(pred.detach(), dim=1)
            cur_pred.extend(pred_result.cpu().tolist())
            val_loss += loss.detach().cpu()
            val_num += data.shape[0]
            valid_acc += (target == pred_result).sum()

    f1score = metrics.f1_score(cur_label, cur_pred, average="macro")
    valid_acc = valid_acc.item() / val_num * 100
    val_loss = val_loss / val_num
    scheduler.step(f1score)
    return f1score, valid_acc, val_loss
