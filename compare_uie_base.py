"""
File Description:
Compare PPaddle UIE-base and Torch UIE-base effect
Author: liushu
Mail: 1554987494@qq.com
Created Time: 2022/11/07
ark_nlp@https://github.com/xiangking/ark-nlp
"""
import os
import sys
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# create paddle uie
from paddlenlp import Taskflow

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
schema = ['时间', '选手', '赛事名称']
_text = "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"
ie = Taskflow('information_extraction', schema=schema)
print(ie(_text))
# create torch uie
from ark_nlp.model.ie.prompt_uie import Tokenizer, PromptUIEConfig, PromptUIE, PromptUIEPredictor

torch_model_path = 'uie-base-torch' # 因为ark库的原因,需要将uie转为bert类型模型
tokenizer = Tokenizer(vocab=torch_model_path, max_seq_len=512) # Paddle UIE-base中默认max_seq_len=512
config = PromptUIEConfig.from_pretrained(torch_model_path)
dl_module = PromptUIE.from_pretrained(torch_model_path, config=config)
ner_predictor_instance = PromptUIEPredictor(dl_module, tokenizer)
entities = []
tmp_entity = {}
for prompt_type in schema:
    for entity in ner_predictor_instance.predict_one_sample([_text, prompt_type]):
        if prompt_type not in tmp_entity:
            tmp_entity[prompt_type] = [{
                'text': entity['entity'],
                'start': entity['start_idx'],
                'end': entity['end_idx'],
            }]
        else:
            tmp_entity[prompt_type].append({
                    'text': entity['entity'],
                    'start': entity['start_idx'],
                    'end': entity['end_idx'],
                })
entities.append(tmp_entity)
print(entities)