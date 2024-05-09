import pandas as pd
import os

from config import CFG


df = pd.read_csv(os.path.join(CFG.data_folder, 'en_train.csv'), usecols=['before', 'after'])
max_length_before = int(df['before'].str.len().max())  # 1057
max_length_after = int(df['after'].str.len().max())  # 3767
