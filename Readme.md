# ERNIE-Pytorch

This project is to convert [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) to [huggingface's](https://github.com/huggingface/pytorch-pretrained-BERT) format.

ERNIE is based on the Bert model and has better performance on Chinese NLP tasks.

## How to use
You can use the version I have converted or convert it by yourself.

### Directly Download

Directly download has converted ERNIE model from [here](http://image.nghuyong.top/ERNIE.zip) and unzip, you will get:

```
ERNIE/
├── bert_config.json
├── pytorch_model.bin
└── vocab.txt
```

### Convert by yourself

1. Download the paddle-paddle version ERNIE model,config and vocab from [here](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) and move to this project path.

2. check the `add_argument` in `convert_ernie_to_pytorch.py` and run `python convert_ernie_to_pytorch.py`, you can get the log:

```
===================extract weights start====================
model config:
attention_probs_dropout_prob: 0.1
hidden_act: relu
hidden_dropout_prob: 0.1
hidden_size: 768
initializer_range: 0.02
max_position_embeddings: 513
num_attention_heads: 12
num_hidden_layers: 12
type_vocab_size: 2
vocab_size: 18000
------------------------------------------------
word_embedding -> bert.embeddings.word_embeddings.weight (18000, 768)
pos_embedding -> bert.embeddings.position_embeddings.weight (513, 768)
sent_embedding -> bert.embeddings.token_type_embeddings.weight (2, 768)
pre_encoder_layer_norm_scale -> bert.embeddings.LayerNorm.gamma (768,)
pre_encoder_layer_norm_bias -> bert.embeddings.LayerNorm.beta (768,)
encoder_layer_0_multi_head_att_query_fc.w_0 -> bert.encoder.layer.0.attention.self.query.weight (768, 768)
encoder_layer_0_multi_head_att_query_fc.b_0 -> bert.encoder.layer.0.attention.self.query.bias (768,)
encoder_layer_0_multi_head_att_key_fc.w_0 -> bert.encoder.layer.0.attention.self.key.weight (768, 768)
encoder_layer_0_multi_head_att_key_fc.b_0 -> bert.encoder.layer.0.attention.self.key.bias (768,)
encoder_layer_0_multi_head_att_value_fc.w_0 -> bert.encoder.layer.0.attention.self.value.weight (768, 768)
encoder_layer_0_multi_head_att_value_fc.b_0 -> bert.encoder.layer.0.attention.self.value.bias (768,)
encoder_layer_0_multi_head_att_output_fc.w_0 -> bert.encoder.layer.0.attention.output.dense.weight (768, 768)
encoder_layer_0_multi_head_att_output_fc.b_0 -> bert.encoder.layer.0.attention.output.dense.bias (768,)
encoder_layer_0_post_att_layer_norm_bias -> bert.encoder.layer.0.attention.output.LayerNorm.bias (768,)
encoder_layer_0_post_att_layer_norm_scale -> bert.encoder.layer.0.attention.output.LayerNorm.weight (768,)
encoder_layer_0_ffn_fc_0.w_0 -> bert.encoder.layer.0.intermediate.dense.weight (3072, 768)
encoder_layer_0_ffn_fc_0.b_0 -> bert.encoder.layer.0.intermediate.dense.bias (3072,)
encoder_layer_0_ffn_fc_1.w_0 -> bert.encoder.layer.0.output.dense.weight (768, 3072)
encoder_layer_0_ffn_fc_1.b_0 -> bert.encoder.layer.0.output.dense.bias (768,)
encoder_layer_0_post_ffn_layer_norm_bias -> bert.encoder.layer.0.output.LayerNorm.bias (768,)
encoder_layer_0_post_ffn_layer_norm_scale -> bert.encoder.layer.0.output.LayerNorm.weight (768,)
.......
encoder_layer_11_multi_head_att_query_fc.w_0 -> bert.encoder.layer.11.attention.self.query.weight (768, 768)
encoder_layer_11_multi_head_att_query_fc.b_0 -> bert.encoder.layer.11.attention.self.query.bias (768,)
encoder_layer_11_multi_head_att_key_fc.w_0 -> bert.encoder.layer.11.attention.self.key.weight (768, 768)
encoder_layer_11_multi_head_att_key_fc.b_0 -> bert.encoder.layer.11.attention.self.key.bias (768,)
encoder_layer_11_multi_head_att_value_fc.w_0 -> bert.encoder.layer.11.attention.self.value.weight (768, 768)
encoder_layer_11_multi_head_att_value_fc.b_0 -> bert.encoder.layer.11.attention.self.value.bias (768,)
encoder_layer_11_multi_head_att_output_fc.w_0 -> bert.encoder.layer.11.attention.output.dense.weight (768, 768)
encoder_layer_11_multi_head_att_output_fc.b_0 -> bert.encoder.layer.11.attention.output.dense.bias (768,)
encoder_layer_11_post_att_layer_norm_bias -> bert.encoder.layer.11.attention.output.LayerNorm.bias (768,)
encoder_layer_11_post_att_layer_norm_scale -> bert.encoder.layer.11.attention.output.LayerNorm.weight (768,)
encoder_layer_11_ffn_fc_0.w_0 -> bert.encoder.layer.11.intermediate.dense.weight (3072, 768)
encoder_layer_11_ffn_fc_0.b_0 -> bert.encoder.layer.11.intermediate.dense.bias (3072,)
encoder_layer_11_ffn_fc_1.w_0 -> bert.encoder.layer.11.output.dense.weight (768, 3072)
encoder_layer_11_ffn_fc_1.b_0 -> bert.encoder.layer.11.output.dense.bias (768,)
encoder_layer_11_post_ffn_layer_norm_bias -> bert.encoder.layer.11.output.LayerNorm.bias (768,)
encoder_layer_11_post_ffn_layer_norm_scale -> bert.encoder.layer.11.output.LayerNorm.weight (768,)
pooled_fc.w_0 -> bert.pooler.dense.weight (768, 768)
pooled_fc.b_0 -> bert.pooler.dense.bias (768,)
====================extract weights done!===================
======================save model start======================
finish save model
finish save config
finish save vocab
======================save model done!======================
```

## Test

```Python
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

"""
indexed_tokens [1, 47, 10, 502, 130, 5, 9150, 10490, 469, 289, 2]
token length 11
last layer shape torch.Size([1, 11, 768]) and value is
tensor([[[ -0.3116,   0.4061,  -0.8272,  ...,  -0.9734,  -1.0078,  -0.3976],
         [  0.7068,  -0.2950,  -0.0637,  ...,  -0.2928,  -0.6499,  -1.8806],
         [  0.1266,   0.0512,   0.4579,  ...,  -0.6819,  -0.5113,  -3.4221],
         ...,
         [  0.6971,  -0.0681,   0.3795,  ...,   0.1983,  -0.3936,  -0.8244],
         [ -0.4181,  -0.3663,   0.4874,  ...,   0.4876,  -0.0783,  -2.6979],
         [ -0.3116,   0.4061,  -0.8272,  ...,  -0.9734,  -1.0078,  -0.3976]]])
"""
```

## Citation

If you use this work in a scientific publication, I would appreciate references to the following BibTex entry:

```latex
@misc{nghuyong2019@ERNIE-Pytorch,
  title={ERNIEPytorch},
  author={Yong Hu},
  howpublished={\url{https://github.com/nghuyong/ERNIE-Pytorch}},
  year={2019}
}
```

## Reference

1. https://arxiv.org/abs/1904.09223
2. https://github.com/PaddlePaddle/LARK/issues/37#issuecomment-474203851

















