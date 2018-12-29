import os
import re
from time import time

import numpy as np
import pypinyin
import tensorflow as tf
from pypinyin import lazy_pinyin

from datasets import audio
from hparams import hparams
from tacotron.synthesizerp import Synthesizer


def pinyin_sentence(sentences):
    pinyin = lazy_pinyin(sentences, style=pypinyin.TONE3)
    return " ".join(pinyin)


taco_checkpoint = os.path.join('logs-Tacotron', 'taco_pretrained/')
checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
synth = Synthesizer()
synth.load(checkpoint_path, hparams)

test_times = 2
wav_time = []
generate_time = []

sentences = [
    '你好','我叫刘新新',
    '这是一条测试数据',
    '中华人民共和国成立了',
    '您的流量还有四十五兆',
    '德国人是很守时的，英国人却不是',
    '昨天我吃了汉堡，今天不想吃了，想吃面',
    '你说的都是真的吗，还是他们说的都是真的。',
    '真是一个奇怪的社会，他们都非常有钱，我却没那么多钱。'
]

dic = list("①①②③④")

for i in range(test_times):
    print("{} time test".format(str(i)))
    for i,sentence in enumerate(sentences):
        start_time = time()
        sentence = pinyin_sentence(sentence)
        sentence = re.sub(r'[a-z]\d', lambda x: x.group(0)[0] + dic[int(x.group(0)[1])], sentence)
        wav = synth.synthesize([sentence])
        output_file = "test-{}.wav".format(str(i))
        audio.save_wav(wav, os.path.join('wav_out/{}'.format(output_file)), sr=hparams.sample_rate)
        stop_time = time()
        one_wav_time = len(wav)/hparams.sample_rate
        one_genrate_time = stop_time - start_time
        print("wav len is {}, generate time is {}".format(str(one_wav_time), str(one_genrate_time)))
        wav_time.append(one_wav_time)
        generate_time.append(one_genrate_time)

wav_time_mean = np.mean(wav_time)
generate_time_mean = np.mean(generate_time)

print("It will comsume time about : {}".format(str(generate_time_mean)))
print("The wav length is {}".format(str(wav_time_mean)))
print("The ratio is {}".format(str(wav_time_mean/generate_time_mean)))
