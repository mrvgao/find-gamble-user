"""
Give the data pipeline iterator.
"""
import tensorflow as tf
from collections import namedtuple


file_name = 'database/train_corpus.txt'

BATCH_SIZE = 32
VOCAB_SIZE = 5822
CATEGORY_SIZE = 2

BatchedInput = namedtuple('BatchedInput', ['train_x', 'train_y', 'initializer'])


def get_train_corps_dict():
    corps_dict = {}
    for line in open(file_name, encoding='utf-8'):
        char = line.split('\t')[0]
        if char not in corps_dict: corps_dict[char] = len(corps_dict)

    return corps_dict


corpus_dict = get_train_corps_dict()


def one_hot(string):
    zeros = [0] * len(corpus_dict)
    categories = [0, 0]
    char = string[0]
    zeros[corpus_dict[char.decode('utf-8')]] = 1
    categories[int((string[1]))] = 1
    return zeros, categories


def get_train_x_y():
    dataset = tf.data.TextLineDataset(filenames=[file_name], buffer_size=10)
    dataset = dataset.map(lambda string: tf.string_split([string], delimiter='\t').values)
    dataset = dataset.map(lambda string: tf.py_func(one_hot, [string], [tf.int64, tf.int64]))
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    train_x, train_y = iterator.get_next()

    return BatchedInput(
        initializer=iterator.initializer,
        train_x=train_x, train_y=train_y
    )


def parse_bytes(ndarrays):
    return [(array[0].decode('utf-8'), array[1].decode('utf-8')) for array in ndarrays]


if __name__ == '__main__':
    batched_input = get_train_x_y()
    initializer = batched_input.initializer
    train_x, train_y = batched_input.train_x, batched_input.train_y

    with tf.Session() as sess:
        sess.run(initializer)
        x, y = sess.run([train_x, train_y])
        print(x)
        print(x[0].shape)

