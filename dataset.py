from torch.utils.data import Dataset
import pandas as pd
import numpy as np


START_TOKEN = '<SRT>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<END>'
english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'X', '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]
word_2_idx = {v: k for k, v in enumerate(english_vocabulary)}
idx_2_word = {k: v for k, v in enumerate(english_vocabulary)}
vocab_size = len(english_vocabulary)
PADDING_IDX = word_2_idx[PADDING_TOKEN]


class TNDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        inputs = list(item['before'])
        targets = list(item['after'])
        input_ids = [word_2_idx[i] for i in inputs]
        target_ids = [word_2_idx[START_TOKEN]] + [word_2_idx[i] for i in targets] + [word_2_idx[END_TOKEN]]
        return input_ids, target_ids


def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_length_input = max([len(inp) for inp in inputs])
    max_length_label = max([len(lab) for lab in labels])
    inputs = [np.pad(inp, (0, max_length_input - len(inp)), 'constant', constant_values=PADDING_IDX)
              for inp in inputs]
    labels = [np.pad(lab, (0, max_length_label - len(lab)), 'constant', constant_values=PADDING_IDX)
              for lab in labels]
    return inputs, labels
