from tacotron.synthesizers import Synthesizer
from hparams import hparams
import tensorflow as tf
from infolog import log
import os


# docker run -p 8500:8500 \
# --mount type=bind,source=/Users/liuxinxin/Documents/GitHub/tacotron/saved_model,target=/models/tacotron \
# -e MODEL_NAME=tacotron -t tensorflow/serving &


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_version", "1",
                    "Model version specify")

taco_checkpoint = os.path.join('logs-Tacotron', 'taco_pretrained/')
checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
synth = Synthesizer()
synth.load(checkpoint_path, hparams,model_version=FLAGS.model_version)

print("Model loaded.")

