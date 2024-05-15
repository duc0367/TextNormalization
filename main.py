import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import TNDataset, collate_fn, vocab_size, PADDING_IDX
from model import Seq2SeqModel
from trainer import train
from utils import seed_everything

seed_everything()

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(os.path.join(CFG.data_folder, 'en_train.csv'), usecols=['before', 'after'])
dataset = TNDataset(df)
dataloader = DataLoader(dataset, batch_size=CFG.batch_size, collate_fn=collate_fn)

model = Seq2SeqModel(vocab_size, CFG.embedding_hidden_size, CFG.hidden_size, device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.learning_rate,
                                                total_steps=CFG.n_epochs * len(dataloader))

model = model.to(device)
train(model, dataloader, criterion, optimizer, scheduler, device)
