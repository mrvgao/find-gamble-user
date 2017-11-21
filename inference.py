"""
Inference a word if is a gamble user's id.

Input: word <string>
Output: the probability of this word to be a gamble user's id.
"""

import tensorflow as tf

sess = tf.Session()

model_path = 'tf-log/-with-loss-0.0066377162002027035.ckpt-71000.meta'
saver = tf.train.import_meta_graph(model_path)
saver.restore(sess, tf.train.latest_checkpoint('tf-log/'))
print(sess.run('embedding/embedding:0'))
