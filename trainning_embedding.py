"""
First, we define the embedding layer, and get each word's embedding.

Second, we define a middle layer, and get the result of E*H, we need E*H to be the if or not is
gamble user.

Third, Save embedding layer.

"""

import tensorflow as tf
from data_pipeline import get_train_x_y
from data_pipeline import convert_word_to_indices
from functools import reduce
import datetime


vocab_size = -1

with open('database/vocab_size.txt') as f:
    vocab_size = int(next(f).split(':')[1])

assert vocab_size > 0
print('vocab size is {}'.format(vocab_size))


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mark', "", 'summary mark')
tf.flags.DEFINE_string('mode', "", 'mode of <train, inference>')


class GambleCharModel:
    def __init__(self, iterator=None):
        self.embedding_size = 10
        self.learning_rate = 1e-3
        self.regularization = 1e-3

        self.__build_layers()

        if iterator:
            self.iterator = iterator
            self.loss, self.op, self.global_step = self.train_batch(
                self.iterator.train_x,
                self.iterator.train_y
            )

    def __build_layers(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(
                'embedding', [vocab_size, self.embedding_size], dtype=tf.float32
            )

        with tf.variable_scope('hidden_layer'):
            self.hidden = tf.get_variable(
                'hidden', [self.embedding_size, 1], initializer=tf.random_normal_initializer, dtype=tf.float32
            )

            self.b = tf.get_variable(
                'hidden_bias', (), initializer=tf.zeros_initializer, dtype=tf.float32
            )

    def run_model(self, x):
        input_embedding = tf.nn.embedding_lookup(self.embedding, x)
        predict = tf.sigmoid(tf.matmul(input_embedding, self.hidden) + self.b)
        # predict \in [0, 1]
        return predict

    def calculate_loss(self, predict, y):
        loss = tf.losses.mean_squared_error(y, predict)
        # loss = tf.losses.absolute_difference(input_y, predict)
        tf.summary.scalar('predication_loss', loss)

        loss += self.regularization * sum([tf.nn.l2_loss(self.embedding),
                                           tf.nn.l2_loss(self.hidden),
                                           tf.nn.l2_loss(self.b)])

        tf.summary.scalar('loss', loss)

        return loss

    def optimize(self, loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   1000, 0.90, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
            .minimize(loss=loss, global_step=global_step)

        return optimizer, global_step

    def train_batch(self, x, y):

        predict = self.run_model(x)
        loss = self.calculate_loss(predict, y)
        optimizer, global_step = self.optimize(loss)

        self.summary = tf.summary.merge_all()

        return loss, optimizer, global_step

    def train(self, sess):
        return sess.run([self.loss, self.op, self.summary, self.global_step])

    def inference(self, sess, x):
        predict = self.run_model(x)
        return sess.run(predict)


def train(session):
    iterator = get_train_x_y(batch_size=256)
    model = GambleCharModel(iterator)
    now = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    summary_writer = tf.summary.FileWriter('tf-log/run-{}-{}'.format(now, FLAGS.mark))

    max_epoch = 10000
    mini_loss = 1e-5

    session.run(iterator.initializer)
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    total_step = 0

    for epoch in range(max_epoch):
        print('epoch --{}--'.format(epoch))
        while True:
            try:
                loss, _, summary, global_step = model.train(sess=session)
                summary_writer.add_summary(summary, global_step=global_step)

                total_step += 1

                if total_step % 10 == 0:
                    print('epoch: {} loss = {}'.format(epoch, loss))

                if total_step % 10 == 0: summary_writer.flush()

                if total_step % 1000 == 0:
                    saver.save(session, 'models/{}-{}-with-loss-{}.ckpt'.format(FLAGS.mark, now, loss), global_step=global_step)

                if loss <= mini_loss:
                    saver.save(session, 'models/last-{}-{}-with-loss-{}.ckpt'.format(FLAGS.mark, now, loss), global_step=global_step)

            except tf.errors.OutOfRangeError:
                epoch += 1
                session.run(iterator.initializer)
                break


def inference(session):
    model_path = 'tf-log/-with-loss-0.0066377162002027035.ckpt-71000.meta'
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(session, tf.train.latest_checkpoint('tf-log/'))

    model = GambleCharModel()
    session.run(tf.global_variables_initializer())

    while True:
        input_string = input('plz input a string(input quit to quit!):')
        if input_string == 'quit': break
        indices = convert_word_to_indices(input_string)
        predication = model.inference(session, indices)
        # combination_probability = 1 - reduce(lambda a, b: a * b, [1 - p for p in predication], 1)
        # 1 - product(1-p_i)
        print(predication)


def main(_):
    session = tf.Session()
    if FLAGS.mode == 'train':
        train(session)
    elif FLAGS.mode == 'inference':
        inference(session)
    else:
        raise TypeError('invalid mode type, must be <train, inference>')


if __name__ == '__main__':
    tf.app.run()
