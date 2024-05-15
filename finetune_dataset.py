import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import CFG


class FinetuneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        inputs = item['before']
        targets = item['after']
        model_inputs = self.tokenizer(inputs, text_target=targets,
                                      max_length=CFG.max_length_encode, truncation=True)
        return model_inputs
