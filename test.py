import os
import re
from time import time

import pypinyin
import tensorflow as tf
from pypinyin import lazy_pinyin

from datasets import audio
from hparams import hparams
from tacotron.synthesizerp import Synthesizer


def pinyin_sentence(sentences):
    pinyin = lazy_pinyin(sentences,style=pypinyin.TONE3)
    return " ".join(pinyin)

taco_checkpoint = os.path.join('logs-Tacotron', 'taco_pretrained/')
checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
synth = Synthesizer()
synth.load(checkpoint_path, hparams)
start = time()
sentences = "我叫做刘新新，是深度学习工程师。"
sentences = pinyin_sentence(sentences)
dic = list("①①②③④")
sentences = re.sub(r'[a-z]\d', lambda x: x.group(0)[0] + dic[int(x.group(0)[1])], sentences)
wav = synth.synthesize([sentences])
output_file = "test.wav"
audio.save_wav(wav, os.path.join('wav_out/{}'.format(output_file)), sr=hparams.sample_rate)
stop = time()
print ("It will cosume time about : {}".format(stop-start))
print ("The wav length is {}".format(str(len(wav)/hparams.sample_rate)))
print ("The ratio is {}".format(str(len(wav)/hparams.sample_rate/(stop-start))))
