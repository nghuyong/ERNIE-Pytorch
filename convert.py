#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
ernie series model conversion based on paddlenlp repository
official repo: https://github.com/PaddlePaddle/ERNIE/blob/repro/README.zh.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD
model list: ernie-1.0, ernie-2.0-en, ernie-2.0-large-en
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2020/7/14
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
        'word_embedding': "bert.embeddings.word_embeddings.weight",
        'pos_embedding': "bert.embeddings.position_embeddings.weight",
        'sent_embedding': "bert.embeddings.token_type_embeddings.weight",
        'pre_encoder_layer_norm_scale': 'bert.embeddings.LayerNorm.gamma',
        'pre_encoder_layer_norm_bias': 'bert.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.w_0'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.b_0'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.w_0'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.b_0'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.w_0'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.b_0'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.w_0'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.b_0'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_scale'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.w_0'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.b_0'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.w_0'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.b_0'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_scale'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooled_fc.w_0': 'bert.pooler.dense.weight',
            'pooled_fc.b_0': 'bert.pooler.dense.bias',
            'mask_lm_trans_fc.w_0': 'cls.predictions.transform.dense.weight',
            'mask_lm_trans_fc.b_0': 'cls.predictions.transform.dense.bias',
            'mask_lm_trans_layer_norm_scale': 'cls.predictions.transform.LayerNorm.gamma',
            'mask_lm_trans_layer_norm_bias': 'cls.predictions.transform.LayerNorm.beta',
            'mask_lm_out_fc.b_0': 'cls.predictions.bias'
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir):
    """
    参数抽取以及转换
    :param input_dir:
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'ernie_config.json'), 'rt', encoding='utf-8'))
    config['layer_norm_eps'] = 1e-5
    config['model_type'] = 'bert'
    config['architectures'] = ["BertForMaskedLM"]  # or 'BertModel'
    if 'sent_type_vocab_size' in config:  # for old version
        config['type_vocab_size'] = config['sent_type_vocab_size']
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
        paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'params'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'w_0' in weight_name:
            weight_value = weight_value.transpose()
        if weight_name not in weight_map:
            print('=' * 20, '[SKIP]', weight_name, '=' * 20)
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    extract_and_convert('./ERNIE_Large_en_stable-2.0.0', './convert')
