#!/usr/bin/env python
# encoding: utf-8
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('./ERNIE')

# Tokenized input
text = "[CLS] 这 是 百度 的 ERNIE 模型 [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

print('indexed_tokens', indexed_tokens)
print('token length', len(indexed_tokens))

segments_ids = [0] * len(indexed_tokens)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('./ERNIE')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

print('last layer shape', encoded_layers[-1].shape, 'and value is ')
print(encoded_layers[-1])
