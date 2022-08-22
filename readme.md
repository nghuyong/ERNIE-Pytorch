# ERNIE-Pytorch

This project is to convert [ERNIE](https://github.com/PaddlePaddle/ERNIE) series models from paddlepaddle
to [huggingface's](https://github.com/huggingface/pytorch-transformers) format (in Pytorch).

## Get Started

Take `ernie-1.0-base-zh` as an example:

```Python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = BertModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### Supported Models

|     Model Name      | Language |           Description           |
|:-------------------:|:--------:|:-------------------------------:|
|  ernie-1.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
|  ernie-2.0-base-en  | English  | Layer:12, Heads:12, Hidden:768  |
| ernie-2.0-large-en  | English  | Layer:24, Heads:16, Hidden:1024 |
|  ernie-3.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
| ernie-3.0-medium-zh | Chinese  |  Layer:6, Heads:12, Hidden:768  |
|  ernie-3.0-mini-zh  | Chinese  |  Layer:6, Heads:12, Hidden:384  |
| ernie-3.0-micro-zh  | Chinese  |  Layer:4, Heads:12, Hidden:384  |
|  ernie-3.0-nano-zh  | Chinese  |  Layer:4, Heads:12, Hidden:312  |
|   ernie-health-zh   | Chinese  | Layer:12, Heads:12, Hidden:768  |
|    ernie-gram-zh    | Chinese  | Layer:12, Heads:12, Hidden:768  |

You can find all the supported models from huggingface's model
hub: [huggingface.co/nghuyong](https://huggingface.co/nghuyong),
and model details from paddle's official
repo: [PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)
and [ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro).

### Note for ERNIE-3.0
If you want to use ernie-3.0 series models, you need to add `task_type_id` to BERT model following this [MR](https://github.com/huggingface/transformers/pull/18686/files) 
**OR** you can re-install the transformers from my changed branch.
```bash
pip uninstall transformers # optional
pip install git+https://github.com/nghuyong/transformers@add_task_type_id # reinstall, 4.22.0.dev0
```
Then you can load ERNIE-3.0 model as before:
```Python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
model = BertModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
```

## Details

<details>
    <summary>I want to convert the model from paddle version by myself 😉</summary>


The following will take `ernie-1.0-base-zh` as an example to show how to convert.

1. Download the paddle-paddle version ERNIE model
   from [here](https://github.com/PaddlePaddle/ERNIE/blob/repro/README.zh.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)
   , move to this project path and unzip the file.
2. ```pip install -r requirements.txt```
3. ```python convert.py```
4. Now, a folder named `convert` will be in the project path, and there will be three files in this
   folder: `config.json`,`pytorch_model.bin` and `vocab.txt`.

</details>

<details>
    <summary>I want to check the calculation results before and after model conversion 😁</summary>

```bash
python test.py --task logit_check
```

You will get the output:

```output
huggingface result
pool output: [-1.         -1.          0.9981035  -0.9996652  -0.78173476 -1.          -0.9994901   0.97012603  0.85954666  0.9854131 ]

paddle result
pool output: [-0.99999976 -0.99999976  0.9981028  -0.9996651  -0.7815545  -0.99999976  -0.9994898   0.97014064  0.8594844   0.985419  ]
```

It can be seen that the result of our convert version is the same with the official paddlepaddle's version.

</details>

<details>
    <summary>I want to reproduce the cloze test in ERNIE1.0's paper 😆</summary>

```bash
python test.py --task cloze_check
```

You will get the output:

```bash
huggingface result
prediction shape:	 torch.Size([47, 18000])
predict result:	 ['西', '游', '记', '是', '中', '国', '神', '魔', '小', '说', '的', '经', '典', '之', '作', '，', '与', '《', '三', '国', '演', '义', '》', '《', '水', '浒', '传', '》', '《', '红', '楼', '梦', '》', '并', '称', '为', '中', '国', '古', '典', '四', '大', '名', '著', '。']
[CLS] logit:	 [-15.693626 -19.522263 -10.429456 ... -11.800728 -12.253127 -14.375117]

paddle result
prediction shape:	 [47, 18000]
predict result:	 ['西', '游', '记', '是', '中', '国', '神', '魔', '小', '说', '的', '经', '典', '之', '作', '，', '与', '《', '三', '国', '演', '义', '》', '《', '水', '浒', '传', '》', '《', '红', '楼', '梦', '》', '并', '称', '为', '中', '国', '古', '典', '四', '大', '名', '著', '。']
[CLS] logit:	 [-15.693538 -19.521954 -10.429307 ... -11.800765 -12.253114 -14.375412]
```

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














