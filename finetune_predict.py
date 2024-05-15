from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

text = '2000'

tokenizer = AutoTokenizer.from_pretrained("text_normalization/checkpoint-8000")
inputs = tokenizer(text, return_tensors="pt").input_ids

model = AutoModelForSeq2SeqLM.from_pretrained("text_normalization/checkpoint-8000")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)
