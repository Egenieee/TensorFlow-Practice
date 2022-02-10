# tensorboard에 text data 표시
import tensorflow as tf

from datetime import datetime
import json
from packaging import version
import tempfile

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# 단일 테스트 기록
my_text = "Hello world! 😃"

# Sets up a timestamped log directory
logdir = "logs/text_basics/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Create a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the text.
with file_writer.as_default():
    tf.summary.text("first_text", my_text, step=0)