from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from config import CFG
from finetune_dataset import FinetuneDataset
from utils import compute_metrics


checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

df = pd.read_csv(os.path.join(CFG.data_folder, 'en_train.csv'))
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = FinetuneDataset(train_df, tokenizer)
val_dataset = FinetuneDataset(val_df, tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir=CFG.output_model,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda x: compute_metrics(x, tokenizer),
)

trainer.train()
