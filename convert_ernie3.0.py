#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
ernie3.0 series model conversion based on paddlenlp repository
official repo: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo
Author: nghuyong liushu
Mail: nghuyong@163.com 1554987494@qq.com
Created Time: 2022/8/17
"""
import collections
import os
import json
import paddle.fluid.dygraph as D
import torch
from paddle import fluid


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'ernie.embeddings.word_embeddings.weight': "bert.embeddings.word_embeddings.weight",
        'ernie.embeddings.position_embeddings.weight': "bert.embeddings.position_embeddings.weight",
        'ernie.embeddings.token_type_embeddings.weight': "bert.embeddings.token_type_embeddings.weight",
        'ernie.embeddings.task_type_embeddings.weight': "bert.embeddings.task_type_embeddings.weight",
        'ernie.embeddings.layer_norm.weight': 'bert.embeddings.LayerNorm.gamma',
        'ernie.embeddings.layer_norm.bias': 'bert.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'ernie.encoder.layers.{i}.linear1.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear1.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.linear2.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear2.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'ernie.pooler.dense.weight': 'bert.pooler.dense.weight',
            'ernie.pooler.dense.bias': 'bert.pooler.dense.bias',
            'cls.predictions.transform.weight': 'cls.predictions.transform.dense.weight',
            'cls.predictions.transform.bias': 'cls.predictions.transform.dense.bias',
            'cls.predictions.layer_norm.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'cls.predictions.layer_norm.bias': 'cls.predictions.transform.LayerNorm.beta',
            'cls.predictions.decoder_bias': 'cls.predictions.bias'
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir):
    """
    抽取并转换
    :param input_dir:
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    if 'init_args' in config:
        config = config['init_args'][0]
    del config['init_class']
    config['layer_norm_eps'] = 1e-5
    config['model_type'] = 'bert'
    config['architectures'] = ["BertForMaskedLM"]  # or 'BertModel'
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'wt', encoding='utf-8'), indent=4)
    print('=' * 20 + 'save vocab file' + '=' * 20)
    with open(os.path.join(input_dir, 'vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
    words = [word.split('\t')[0] for word in words]
    with open(os.path.join(output_dir, 'vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word + "\n")
    print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'ernie_3.0_base_zh.pdparams'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'weight' in weight_name:
            if 'ernie.encoder' in weight_name or 'ernie.pooler' in weight_name or 'cls.' in weight_name:
                weight_value = weight_value.transpose()
        if weight_name not in weight_map:
            print('=' * 20, '[SKIP]', weight_name, '=' * 20)
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    from paddlenlp.transformers import ErnieForMaskedLM, ErnieTokenizer

    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-base-zh")
    model = ErnieForMaskedLM.from_pretrained("ernie-3.0-base-zh")
    extract_and_convert('/Users/huyong/.paddlenlp/models/ernie-3.0-base-zh/', './convert/')
