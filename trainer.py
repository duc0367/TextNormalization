import torch
import time

from config import CFG
from AverageMeter import AverageMeter
from utils import time_since
from dataset import word_2_idx, idx_2_word

validation_text = '1988'


def validate(model, device):
    model.eval()
    validation_ids = [word_2_idx[word] for word in list(validation_text)]

    inputs = torch.tensor(validation_ids).to(device)
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(inputs)   # (L,)
    words = [idx_2_word[i] for i in outputs]
    prediction = ''.join(words)
    print(f'Prediction of {validation_text}: {prediction}')
    model.train()


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_autocast)
    losses = AverageMeter()
    start = time.time()

    for epoch in range(CFG.n_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=CFG.use_autocast):
                outputs = model(inputs, labels)  # (N, L, H)
                outputs = outputs[:, 1:, :].view(-1, outputs.shape[-1])  # (N * L, H)
                labels = labels[:, 1:].view(-1)  # (N * L)
                loss = criterion(outputs, labels)
            losses.update(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (batch_idx + 1) % CFG.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] '
                      'Loss: {loss:.4f} '
                      'Elapsed Time: {remain:s} '
                      'Grad Norm: {grad_norm.val:.4f}'
                      .format(epoch + 1, batch_idx + 1, len(train_loader),
                              loss=loss,
                              remain=time_since(start, float(batch_idx + 1) / len(train_loader)),
                              grad_norm=grad_norm))
