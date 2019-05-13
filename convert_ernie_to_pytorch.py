#!/usr/bin/env python
# encoding: utf-8
import collections
import os
import sys
import numpy as np
import argparse
import paddle.fluid as fluid
import torch
import json

if not os.path.exists('LARK'):
    os.system('git clone https://github.com/PaddlePaddle/LARK.git')
sys.path = ['./LARK/ERNIE'] + sys.path
try:
    from model.ernie import ErnieConfig
    from finetune.classifier import create_model
except:
    raise Exception('Place clone ERNIE first')


def if_exist(var):
    return os.path.exists(os.path.join(args.init_pretraining_params, var.name))


def build_weight_map():
    weight_map = collections.OrderedDict({
        'word_embedding': 'bert.embeddings.word_embeddings.weight',
        'pos_embedding': 'bert.embeddings.position_embeddings.weight',
        'sent_embedding': 'bert.embeddings.token_type_embeddings.weight',
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


def extract_weights(args):
    # add ERNIR to environment
    print('extract weights start'.center(60, '='))
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    args.max_seq_len = 512
    args.use_fp16 = False
    args.num_labels = 2
    args.loss_scaling = 1.0
    print('model config:')
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            _, _ = create_model(
                args,
                pyreader_name='train',
                ernie_config=ernie_config)
    fluid.io.load_vars(exe, args.init_pretraining_params, main_program=test_prog, predicate=if_exist)
    state_dict = collections.OrderedDict()
    weight_map = build_weight_map()
    for ernie_name, pytorch_name in weight_map.items():
        fluid_tensor = fluid.global_scope().find_var(ernie_name).get_tensor()
        fluid_array = np.array(fluid_tensor, dtype=np.float32)
        if 'w_0' in ernie_name:
            fluid_array = fluid_array.transpose()
        state_dict[pytorch_name] = fluid_array
        print(f'{ernie_name} -> {pytorch_name} {fluid_array.shape}')
    print('extract weights done!'.center(60, '='))
    return state_dict


def save_model(state_dict, dump_path):
    print('save model start'.center(60, '='))
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    # save model
    for key in state_dict:
        state_dict[key] = torch.FloatTensor(state_dict[key])
    torch.save(state_dict, os.path.join(dump_path, "pytorch_model.bin"))
    print('finish save model')
    # save config
    ernie_config = ErnieConfig(args.ernie_config_path)._config_dict
    # set layer_norm_eps, more detail see: https://github.com/PaddlePaddle/LARK/issues/75
    ernie_config['layer_norm_eps'] = 1e-5
    with open(os.path.join(dump_path, "bert_config.json"), 'wt', encoding='utf-8') as f:
        json.dump(ernie_config, f, indent=4)
    print('finish save config')
    # save vocab.txt
    vocab_f = open(os.path.join(dump_path, "vocab.txt"), "wt", encoding='utf-8')
    with open("./LARK/ERNIE/config/vocab.txt", "rt", encoding='utf-8') as f:
        for line in f:
            data = line.strip().split("\t")
            vocab_f.writelines(data[0] + "\n")
    vocab_f.close()
    print('finish save vocab')
    print('save model done!'.center(60, '='))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_pretraining_params", default='./ERNIE_stable-1.0.1.tar/params', type=str, help=".")
    parser.add_argument("--ernie_config_path", default='./ERNIE_stable-1.0.1.tar/ernie_config.json', type=str, help=".")
    parser.add_argument("--output_dir", default='./ERNIE', type=str, help=".")
    args = parser.parse_args()
    state_dict = extract_weights(args)
    save_model(state_dict, args.output_dir)
