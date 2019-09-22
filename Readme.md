# ERNIE-Pytorch

This project is to convert [ERNIE](https://github.com/PaddlePaddle/ERNIE) to [pytorch-transformers's](https://github.com/huggingface/pytorch-transformers) format.

ERNIE is based on the Bert model and has better performance on Chinese NLP tasks.

**Currently this project only supports the conversion of ERNIE 1.0 version.**

## How to use
You can use the version I have converted or convert it by yourself.

requirements

```txt
paddlepaddle-gpu==1.4.0.post87
```

### Directly Download

Directly download has converted ERNIE model:

|model|description|
|:---:|:---:|
|[ERNIE 1.0 Base for Chinese](https://drive.google.com/open?id=1k7G41gaQvaqOhmQt-b5KSj27YcHjdSpV)|with params, config and vocabs|
|[ERNIE 1.0 Base for Chinese(max-len-512)](https://drive.google.com/open?id=1il88pC5DabgypSYAF8pq_E2cuNrNuUAC)|with params, config and vocabs|

### Convert by yourself

1. Download the paddle-paddle version ERNIE1.0 model from [here](https://github.com/PaddlePaddle/ERNIE#models), and move to this project path.

2. check the `add_argument` in `convert_ernie_to_pytorch.py` and run `python convert_ernie_to_pytorch.py`, you can get the log:

```
===================extract weights start====================
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

### Obtain the parameters for Mask-LM task

Run `python convert_ernie_to_pytorch_v2.py` (which is modified from `convert_ernie_to_pytorch.py`), it will save the following parameters extra

```yml
{
        'mask_lm_trans_fc.b_0': 'cls.predictions.transform.dense.bias',
        'mask_lm_trans_fc.w_0': 'cls.predictions.transform.dense.weight',
        'mask_lm_trans_layer_norm_scale': 'cls.predictions.transform.LayerNorm.weight',
        'mask_lm_trans_layer_norm_bias': 'cls.predictions.transform.LayerNorm.bias',
        'mask_lm_out_fc.b_0': 'cls.predictions.bias'
}
```

You can use `BertForMaskedLM` from [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) to test the converted model, an example is shown below, where bert-base is google's Chinese-BERT, bert-wwm and bert-wwm-ext are download from [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).
```yml
input: [MASK] [MASK] [MASK] 是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名。
output:
{
        "bert-base": "《 神 》",
        "bert-wwm": "天 神 奇",
        "bert-wwm-ext": "西 游 记",
        "ernie": "西 游 记"
}
```


## Test

```Python
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

















