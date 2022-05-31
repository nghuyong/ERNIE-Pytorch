#!/usr/bin/env python
# encoding: utf-8
"""
File Description: 对于老版ernie(paddle1.x版本)的转换
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


def build_params_map():
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

    def add_w_and_b(ernie_pre, pytroch_pre):
        weight_map[ernie_pre + ".w_0"] = pytroch_pre + ".weight"
        weight_map[ernie_pre + ".b_0"] = pytroch_pre + ".bias"

    def add_one_encoder_layer(layer_number):
        # attention
        add_w_and_b(f"encoder_layer_{layer_number}_multi_head_att_query_fc",
                    f"bert.encoder.layer.{layer_number}.attention.self.query")
        add_w_and_b(f"encoder_layer_{layer_number}_multi_head_att_key_fc",
                    f"bert.encoder.layer.{layer_number}.attention.self.key")
        add_w_and_b(f"encoder_layer_{layer_number}_multi_head_att_value_fc",
                    f"bert.encoder.layer.{layer_number}.attention.self.value")
        add_w_and_b(f"encoder_layer_{layer_number}_multi_head_att_output_fc",
                    f"bert.encoder.layer.{layer_number}.attention.output.dense")
        weight_map[f"encoder_layer_{layer_number}_post_att_layer_norm_bias"] = \
            f"bert.encoder.layer.{layer_number}.attention.output.LayerNorm.bias"
        weight_map[f"encoder_layer_{layer_number}_post_att_layer_norm_scale"] = \
            f"bert.encoder.layer.{layer_number}.attention.output.LayerNorm.weight"
        # intermediate
        add_w_and_b(f"encoder_layer_{layer_number}_ffn_fc_0", f"bert.encoder.layer.{layer_number}.intermediate.dense")
        # output
        add_w_and_b(f"encoder_layer_{layer_number}_ffn_fc_1", f"bert.encoder.layer.{layer_number}.output.dense")
        weight_map[f"encoder_layer_{layer_number}_post_ffn_layer_norm_bias"] = \
            f"bert.encoder.layer.{layer_number}.output.LayerNorm.bias"
        weight_map[f"encoder_layer_{layer_number}_post_ffn_layer_norm_scale"] = \
            f"bert.encoder.layer.{layer_number}.output.LayerNorm.weight"

    for i in range(12):
        add_one_encoder_layer(i)
    add_w_and_b('pooled_fc', 'bert.pooler.dense')
    return weight_map


def extract_and_convert(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'ernie_config.json'), 'rt', encoding='utf-8'))
    config['layer_norm_eps'] = 1e-5
    if 'sent_type_vocab_size' in config:
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
    weight_map = build_params_map()
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
    extract_and_convert('./eHealth-base-discriminator', './convert')
