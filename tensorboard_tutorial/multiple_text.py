import tensorflow as tf

from datetime import datetime
import json
from packaging import version
import tempfile

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

logdir = "logs/multiple_texts/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
    with tf.name_scope("name_scope_1"):
        for step in range(20):
            tf.summary.text("a_stream_of_text", f"Hello from step {step}", step=step)
            tf.summary.text("another_stream_of_text", f"This can be kept separate {step}", step=step)
    with tf.name_scope("name_scope_2"):
        tf.summary.text("just_from_step_0", "This is an important announcement from step 0", step=0)