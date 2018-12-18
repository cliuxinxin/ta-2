import argparse
import os
import sox
from datetime import datetime
import pypinyin
from pypinyin import lazy_pinyin
from flask_cors import CORS

from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
import tensorflow as tf
import json

from hparams import hparams
# from tacotron.synthesize import tacotron_synthesize

import re
import time
from time import sleep

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
# from tacotron.synthesizer import Synthesizer
from tacotron.synthesizerp import Synthesizer
from tqdm import tqdm
from time import time
import numpy as np
import redis
import random
from datasets import audio
from threading import Thread


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    taco_checkpoint = os.path.join('logs-' + args.model, 'taco_' + args.checkpoint)

    return taco_checkpoint, modified_hp




def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    # Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    # Set inputs batch wise
    sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
                 range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(eval_dir))
    return eval_dir

def run_server(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')
    file_names = []

    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    # Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    # Set inputs batch wise
    sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
                 range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
            filename = synth.synthesize(texts, basenames, None)
            file_names.append(filename)

    file_names = " ".join(file_names)

    log('files generate at wav_out {}'.format(file_names))
    return eval_dir


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
    output_dir = 'tacotron_' + args.output_dir

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
        raise ValueError(
            'Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
                hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
        raise ValueError(
            'Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
                hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if args.mode == 'eval':
        return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
    else:
        return run_server(args, checkpoint_path, output_dir, hparams, sentences)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    accepted_modes = ['eval', 'synthesis', 'live']
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='Tacotron')
    parser.add_argument('--mels_dir', default='tacotron_output/eval/',
                        help='folder to contain mels to synthesize audio from using the Wavenet')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='live', help='mode of run: can be one of {}'.format(accepted_modes))
    args = parser.parse_args()

    taco_checkpoint, hparams = prepare_run(args)
    sentences = hparams.sentences

    _ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)


#
# if __name__ == '__main__':
#     main()

def pinyin_sentence(sentences):
    pinyin = lazy_pinyin(sentences,style=pypinyin.TONE3)
    return " ".join(pinyin)

app = Flask(__name__)
app.debug = False
CORS(app, supports_credentials=True)

# accepted_modes = ['eval', 'synthesis', 'live']
# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
# parser.add_argument('--hparams', default='',
#                         help='Hyperparameter overrides as a comma-separated list of name=value pairs')
# parser.add_argument('--model', default='Tacotron')
# parser.add_argument('--mels_dir', default='tacotron_output/eval/',
#                         help='folder to contain mels to synthesize audio from using the Wavenet')
# parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
# parser.add_argument('--mode', default='live', help='mode of run: can be one of {}'.format(accepted_modes))
# args = parser.parse_args()

# taco_checkpoint, hparams = prepare_run(args)

taco_checkpoint = os.path.join('logs-Tacotron', 'taco_pretrained/')

checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
synth = Synthesizer()
synth.load(checkpoint_path, hparams)
basenames='base'
r = redis.Redis(host='localhost', port=6379, decode_responses=True,db=1)

class SaveWav(Thread):
    def __init__(self, wav,record_sentences):
        Thread.__init__(self)
        self.wav = wav
        self.record_sentences = record_sentences

    def run(self):
        nowTime = datetime.now()
        output_file = 'wav-' + nowTime.strftime('%Y%m%d%H%M%S') + str(nowTime.microsecond) + '.npy'
        np.save(os.path.join('wav_out/{}'.format(output_file)), self.wav, allow_pickle=False)
        r.set(self.record_sentences, output_file)

@app.route('/', methods=['get'])
def index():
    dic = list("①①②③④")
    dict = {}
    start = time()
    sentences = request.args.get('sentences')
    record_sentences = sentences
    if r.get(record_sentences):
        file_name = r.get(record_sentences)
        wav = np.load(os.path.join('wav_out/{}'.format(file_name)))
    else:
        sentences = pinyin_sentence(sentences)
        sentences = re.sub(r'[a-z]\d', lambda x: x.group(0)[0] + dic[int(x.group(0)[1])], sentences)
        sentences = sentences.split("。")
        while '' in sentences:
            sentences.remove('')
        wav= synth.synthesize(sentences)
        thread_a = SaveWav(wav,record_sentences)
        thread_a.start()

    stop = time()
    dict['wav_file'] = wav
    dict['time'] = stop-start
    return json.dumps(dict, cls=NumpyEncoder)

#
if __name__ == "__main__":
    server = pywsgi.WSGIServer(('127.0.0.1', 19887), app)
    server.serve_forever()
    # app.run(host='0.0.0.0:19877')


