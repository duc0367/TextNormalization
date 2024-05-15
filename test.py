from transformers import AutoTokenizer

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(tokenizer.cls_token_id, tokenizer.cls_token, tokenizer.eos_token_id)
print(tokenizer('hello world'))
