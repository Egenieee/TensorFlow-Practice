import tensorflow as tf

from datetime import datetime
import json
from packaging import version
import tempfile

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

logdir = "logs/markdown/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

some_obj_worth_noting = {
    "tfds_training_data" : {
        "name" : "mnist",
        "split" : "train",
        "shuffle_files" : "True",
    },
    "keras_optimizer" : {
        "name" : "Adagrad",
        "learning_rate" : "0.001",
        "epsilon" : 1e-7
    },
    "hardward" : "Cloud TPU"
}

# TODO: Update this example when TensorBoard is released with
# https://github.com/tensorflow/tensorboard/pull/4585
# which supports fenced codeblocks in Markdown.
def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))

markdown_text = """
# 글머리 대빵
## 글머리 대빵 오른팔
### 글머리 대빵 왼팔

***

기본적인 단락

    this is a code block.

코드 블럭 끝

***

_이탤릭체_
**볼드체**

나는 **어제** _굽네치킨_ 을 먹었다,

***

[Google](https://google.com, "google link")

***

| and | tables |
| ---- | ---------- |
| among | others |

***

1. 첫번째
2. 두번째
3. 세번째

***

> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.

"""

with file_writer.as_default():
  tf.summary.text("run_params", pretty_json(some_obj_worth_noting), step=0)
  tf.summary.text("markdown_jubiliee", markdown_text, step=0)