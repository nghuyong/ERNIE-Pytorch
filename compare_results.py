#!/usr/bin/env python
# encoding: utf-8
"""
File Description: 
Author: liushu
Mail: 1554987494@qq.com
Created Time: 2022/8/17
"""
#%%
# torch
import os
import sys
import torch
from transformers import BertModel, BertTokenizer
# from bert_ernie3.modeling_bert import BertModel # 带task type embedding的BertModel
tokenizer = BertTokenizer.from_pretrained('./ernie-3.0-base-zh-torch')
model = BertModel.from_pretrained('./ernie-3.0-base-zh-torch')
input_ids = torch.tensor([tokenizer.encode(text="你好",add_special_tokens=True)])
print('input_ids:', input_ids)
model.eval()
with torch.no_grad():
    pooled_output = model(input_ids)[1]
    print(pooled_output.numpy())
#%%
# paddle
import paddle
import paddlenlp
from paddlenlp.transformers import ErnieModel
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
model = paddlenlp.transformers.ErnieModel.from_pretrained("ernie-3.0-base-zh", use_task_id=False)
# input_ids = paddle.tensor([tokenizer.encode(text="你好",add_special_tokens=True)])
inputs = tokenizer("你好")
inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
# inputs['use_task_id'] = False
print(inputs)
model.eval()
with paddle.no_grad():
    sequence_output, pooled_output = model(**inputs)
    print(pooled_output.numpy())