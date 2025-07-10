from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("pretrain_models/bert-base-uncased", use_fast=True)  # 可以使用use fast加速

print("sep token id: ", tokenizer.sep_token_id)

sequence = "Using a Transformer network is simple"

# Encoding
print('Encoding')

# 1. tokenization: 对文本进行分词
tokens = tokenizer.tokenize(sequence)

print("toekns: ", tokens)

# 2. convert_tokens_to_ids：将分词后的token 映射为数字
ids = tokenizer.convert_tokens_to_ids(tokens)

print("idx: ", ids)

print("encode: ", tokenizer.encode(sequence))
print("encode(add_special_tokens=False): ",
      tokenizer.encode(sequence, add_special_tokens=False))

# Decoding
print('Decoding')

# Decoding 的作用是将输出的 ids 转化为文本
decoded_string = tokenizer.decode(ids)

print("decode string: ", decoded_string)
