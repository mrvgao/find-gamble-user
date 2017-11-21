"""
Give the data pipeline iterator.
"""
import tensorflow as tf
from collections import namedtuple


file_name = 'database/train_corpus.tsv'

VOCAB_SIZE = 6859

BatchedInput = namedtuple('BatchedInput', ['train_x', 'train_y', 'initializer'])


def get_train_characters():
    characters = []
    line_num = 0
    for ii, line in enumerate(open(file_name, encoding='utf-8')):
        if ii == 0: continue

        char = line.split('\t')[0]
        line_num += 1
        characters.append(char)

    return characters


all_characters = get_train_characters()


def get_encoding(string):
    probability = float(string[1])
    char = string[0]
    global all_characters
    return all_characters.index(char.decode('utf-8')), probability


def convert_word_to_indices(words):
    global all_characters
    return [all_characters.index(c) for c in words]


def get_train_x_y(batch_size=128):
    dataset = tf.data.TextLineDataset(filenames=[file_name], buffer_size=10).skip(1)
    dataset = dataset.map(lambda string: tf.string_split([string], delimiter='\t').values)
    dataset = dataset.map(lambda string: tf.py_func(get_encoding, [string], [tf.int64, tf.float64]))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(batch_size)
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
    src_train_x, src_train_y = batched_input.train_x, batched_input.train_y

    with tf.Session() as sess:
        sess.run(initializer)
        x, y = sess.run([src_train_x, src_train_y])
        print(x)
        print(y)
        print(x[0].shape)

