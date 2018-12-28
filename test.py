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
test_times = 10
wav_time = []
generate_time = []
for i in range(test_times):
    print("{} time test".format(str(i)))
    start = time()
    sentences = "我叫刘新新"
    sentences = pinyin_sentence(sentences)
    dic = list("①①②③④")
    sentences = re.sub(r'[a-z]\d', lambda x: x.group(0)[0] + dic[int(x.group(0)[1])], sentences)
    wav = synth.synthesize([sentences])
    output_file = "test.wav"
    audio.save_wav(wav, os.path.join('wav_out/{}'.format(output_file)), sr=hparams.sample_rate)
    stop = time()
    one_wav_time = len(wav)/hparams.sample_rate
    one_genrate_time = stop - start
    print("wav len is {}, generate time is {}".format(str(one_wav_time), str(one_genrate_time)))
    wav_time.append(one_wav_time)
    generate_time.append(one_genrate_time)

wav_time_mean = np.mean(wav_time)
generate_time_mean = np.mean(generate_time)
print("It will cosume time about : {}".format(str(generate_time_mean)))
print("The wav length is {}".format(str(wav_time_mean)))
print("The ratio is {}".format(str(wav_time_mean/generate_time_mean)))
