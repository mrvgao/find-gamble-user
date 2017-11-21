"""
First, we define the embedding layer, and get each word's embedding.

Second, we define a middle layer, and get the result of E*H, we need E*H to be the if or not is
gamble user.

Third, Save embedding layer.

"""

import tensorflow as tf
from data_pipeline import get_train_x_y
import datetime


vocab_size = -1

with open('database/vocab_size.txt') as f:
    vocab_size = int(next(f).split(':')[1])

assert vocab_size > 0
print('vocab size is {}'.format(vocab_size))


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mark', "", 'summary mark')


class GambleCharModel:
    def __init__(self, iterator):
        self.iterator = iterator
        self.embedding_size = 10
        self.learning_rate = 1e-3
        self.regularization = 1e-3

        self.__build_layers()
        self.loss, self.op = self.__build_model()

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

    def __build_model(self):
        input_x = self.iterator.train_x
        self.input_embedding = tf.nn.embedding_lookup(self.embedding, input_x)
        predict = tf.sigmoid(tf.matmul(self.input_embedding, self.hidden) + self.b)
        # predict \in [0, 1]

        input_y = self.iterator.train_y
        loss = tf.losses.mean_squared_error(input_y, predict)
        # loss = tf.losses.absolute_difference(input_y, predict)

        tf.summary.scalar('predication_loss', loss)

        loss += self.regularization * sum([tf.nn.l2_loss(self.embedding),
                                           tf.nn.l2_loss(self.hidden),
                                           tf.nn.l2_loss(self.b)])

        starter_learning_rate = self.learning_rate

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   1000, 0.90, staircase=False)

        tf.summary.scalar('global_step', self.global_step)
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(loss=loss, global_step=self.global_step)

        self.summary = tf.summary.merge_all()

        return loss, optimizer

    def train(self, sess):
        return sess.run([self.loss, self.op, self.summary, self.global_step])


def main(_):
    iterator = get_train_x_y(batch_size=256)
    model = GambleCharModel(iterator)

    session = tf.Session()
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


if __name__ == '__main__':
    tf.app.run()
