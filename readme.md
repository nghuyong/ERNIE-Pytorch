# ERNIE-Pytorch

This project is to convert [ERNIE](https://github.com/PaddlePaddle/ERNIE) series models from paddlepaddle to [huggingface's](https://github.com/huggingface/pytorch-transformers) format (in Pytorch).

## Get Started
Take `ernie-1.0` as an example:
```Python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-1.0")
model = BertModel.from_pretrained("nghuyong/ernie-1.0")
```

### Supported Models
|Model|Language|Identifier in huggingface|Description|
|:---:|:---:|:---:|:---:|
|ernie-1.0|Chinese|nghuyong/ernie-1.0|Layer:12, Hidden:768, Heads:12|
|ernie-tiny|English|nghuyong/ernie-tiny|Layer:3, Hdden:1024, Heads:16|
|ernie-2.0-en|English|nghuyong/ernie-2.0-en|Layer:12, Hidden:768, Heads:12|
|ernie-2.0-large-en|English|nghuyong/ernie-2.0-large-en|Layer:24, Hidden:1024, Heads:16|
|ernie-gram-zh|Chinese|nghuyong/ernie-gram-zh|Layer:12, Hidden:768, Heads:12|
|ernie-health-zh|Chinese|nghuyong/ernie-health-zh|Layer:12, Hidden:768, Heads:12|

You can find all the supported models from huggingface's model hub: https://huggingface.co/nghuyong, 
and model details from ERNIE's official repo: https://github.com/PaddlePaddle/ERNIE.

## Details
<details>
    <summary>I want to convert the model from paddle version by myself 😉</summary>


Cool!!! The following will take `ernie-1.0` as an example to show how to convert.

1. Download the paddle-paddle version ERNIE model from [here](https://github.com/PaddlePaddle/ERNIE#3-%E4%B8%8B%E8%BD%BD%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%8F%AF%E9%80%89), move to this project path and unzip the file.
2. ```pip install -r requirements.txt```
3. ```python convert.py```
4. Now, a folder named `convert` will be in the project path, and there will be three files in this folder: `config.json`,`pytorch_model.bin` and `vocab.txt`.
</details>

<details>
    <summary>I want to check the calculation results before and after model conversion 😁</summary>


[PaddlePaddle's Official Quick Start](https://github.com/PaddlePaddle/ERNIE#%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B)
```Python
#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel

D.guard().__enter__() # activate paddle `dygrpah` mode

model = ErnieModel.from_pretrained('ernie-1.0')    # Try to get pretrained model from server, make sure you have network connection
model.eval()
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

ids, _ = tokenizer.encode('hello world')
ids = D.to_variable(np.expand_dims(ids, 0))  # insert extra `batch` dimension
pooled, encoded = model(ids)                 # eager execution
print(pooled.numpy())                        # convert  results to numpy

"""
output:
[[-1.         -1.          0.99479663 -0.99986964 -0.7872066  -1.
  -0.99919444  0.985997   -0.22648102  0.97202295 -0.9994965  -0.982234
  -0.6821966  -0.9998574  -0.83046496 -0.9804977  -1.          0.9999509
  -0.55144966  0.48973152 -1.          1.          0.14248642 -0.71969527
   ...
   0.93848914  0.8418771   1.          0.99999803  0.9800671   0.99886674
   0.9999988   0.99946415  0.9849099   0.9996924  -0.79442227 -0.9999412
   0.99827075  1.         -0.05767363  0.99999857  0.8176171   0.7983498
  -0.14292054  1.         -0.99759513 -0.9999982  -0.99973375 -0.9993742 ]]
"""
```

Use huggingface's Transformer with our converted ERNIE model
````Python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('nghuyong/ernie-1.0')
model = BertModel.from_pretrained('nghuyong/ernie-1.0')
input_ids = torch.tensor([tokenizer.encode("hello world", add_special_tokens=True)])
with torch.no_grad():
    pooled_output = model(input_ids)[1]
    print(pooled_output.numpy())

"""
output:
[[-1.         -1.          0.99479663 -0.99986964 -0.78720796 -1.
  -0.9991946   0.98599714 -0.22648017  0.972023   -0.9994966  -0.9822342
  -0.682196   -0.9998575  -0.83046496 -0.9804982  -1.          0.99995095
  -0.551451    0.48973027 -1.          1.          0.14248991 -0.71969616
   ...
   0.9384899   0.84187615  1.          0.999998    0.9800671   0.99886674
   0.9999988   0.99946433  0.98491037  0.9996923  -0.7944245  -0.99994105
   0.9982707   1.         -0.05766615  0.9999987   0.81761867  0.7983511
  -0.14292456  1.         -0.9975951  -0.9999982  -0.9997338  -0.99937415]]
"""
````
It can be seen that the encoder result of our convert version is the same with the official paddlepaddle's version. 

</details>

<details>
    <summary>I want to reproduce the Close Test in ERNIE's paper 😆</summary>


We will compare ERNIE's result with google's Chinese-BERT, bert-wwm and bert-wwm-ext from [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

```Python
#!/usr/bin/env python
#encoding: utf-8
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('nghuyong/ernie-1.0')

input_tx = "[CLS] [MASK] [MASK] [MASK] 是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。[SEP]"
tokenized_text = tokenizer.tokenize(input_tx)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([[0] * len(tokenized_text)])

model = BertForMaskedLM.from_pretrained('nghuyong/ernie-1.0')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0, (len(tokenized_text) - 1))]
predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in
                   range(1, (len(tokenized_text) - 1))]

print('Predicted token is:', predicted_token)
```

Result
```Latext
input:
[CLS] [MASK] [MASK] [MASK] 是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。[SEP]
output:
{
    "bert-base": "《 神 》",
    "bert-wwm": "天 神 奇",
    "bert-wwm-ext": "西 游 记",
    "ernie-1.0": "西 游 记"
}
```
</details>

<details>
<summary>I also want a Tensorflow's version 🙃</summary>


We can simply use huggingface's [convert_pytorch_checkpoint_to_tf](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_pytorch_checkpoint_to_original_tf.py) tool to
convert huggingface's pytorch model to tensorflow's version.

```Python
from transformers import BertModel
from transformers.convert_bert_pytorch_checkpoint_to_original_tf import convert_pytorch_checkpoint_to_tf

model = BertModel.from_pretrained('nghuyong/ernie-1.0')
convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir='./tf_convert', model_name='ernie')
```

The above code will generate a `tf_convert` directory with tensorflow's checkpoint.
```bash
└── tf_convert
    ├── checkpoint
    ├── ernie.ckpt.data-00000-of-00001
    ├── ernie.ckpt.index
    └── ernie.ckpt.meta
```
The `config.json` and `vocab.txt` of tensorflow version is the same with huggingface's pytorch version in `convert` directory.

</details>


## Citation

If you use this work in a scientific publication, I would appreciate that you can also cite the following BibTex entry:

```latex
@misc{nghuyong2019@ERNIE-Pytorch,
  title={ERNIEPytorch},
  author={Yong Hu},
  howpublished={\url{https://github.com/nghuyong/ERNIE-Pytorch}},
  year={2019}
}
```














