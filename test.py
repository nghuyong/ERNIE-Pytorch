#!/usr/bin/env python
# encoding: utf-8
import torch
from pytorch_transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('./ERNIE-converted')

input_ids = torch.tensor([tokenizer.encode("这是百度的ERNIE1.0模型")])

model = BertModel.from_pretrained('./ERNIE-converted')

all_hidden_states, all_attentions = model(input_ids)[-2:]

print('all_hidden_states shape', all_hidden_states.shape)
print(all_hidden_states)
"""
all_hidden_states shape torch.Size([1, 12, 768])
tensor([[[-0.2229, -0.3131,  0.0088,  ...,  0.0199, -1.0507,  0.5315],
         [-0.8425, -0.0086,  0.2039,  ..., -0.1681,  0.0459, -1.1015],
         [ 0.7147,  0.1788,  0.7055,  ...,  0.4651,  0.8798, -0.5982],
         ...,
         [-0.9507, -0.3732, -0.9508,  ...,  0.4992, -0.0545,  1.2238],
         [ 0.2940,  0.0286, -0.2381,  ...,  1.0630,  0.0387, -0.5267],
         [-0.1940,  0.1136,  0.0118,  ...,  0.9859,  0.4807, -1.5650]]],
       grad_fn=<NativeLayerNormBackward>)
"""
