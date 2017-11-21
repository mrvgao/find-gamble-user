"""
First, we define the embedding layer, and get each word's embedding.

Second, we define a middle layer, and get the result of E*H, we need E*H to be the if or not is
gamble user.

Third, Save embedding layer.

"""

import tensorflow as tf
from data_pipeline import get_train_x_y
import datetime


vocab_size = 5822

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mark', "", 'summary mark')


class GambleCharModel:
    def __init__(self, iterator):
        self.iterator = iterator
        self.embedding_size = 5
        self.learning_rate = 1e-3

        self.__build_layers()
        self.loss, self.op = self.__build_model()

    def __build_layers(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(
                'embedding', [vocab_size, self.embedding_size], dtype=tf.float64
            )

        with tf.variable_scope('hidden_layer'):
            self.hidden = tf.get_variable(
                'hidden', [self.embedding_size, 2], initializer=tf.random_normal_initializer, dtype=tf.float64
            )

            self.b = tf.get_variable(
                'hidden_bias', (), initializer=tf.zeros_initializer, dtype=tf.float64
            )

    def __build_model(self):
        input_x = self.iterator.train_x
        self.input_embedding = tf.nn.embedding_lookup(self.embedding, input_x)
        predict = tf.matmul(self.input_embedding, self.hidden) + self.b
        # output size == (None, 2)

        input_y = self.iterator.train_y
        loss = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=predict)
        loss += tf.nn.l2_loss(self.embedding)

        starter_learning_rate = self.learning_rate

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   1000, 0.90, staircase=False)

        tf.summary.scalar('global_step', self.global_step)
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

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

    max_epoch = 100

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
                    print('loss = {}'.format(loss))

                if global_step % 10 == 0: summary_writer.flush()

                if global_step % 1000 == 0:
                    saver.save(session, 'models/{}-{}-with-loss-{}'.format(FLAGS.mark, now, loss))

            except tf.errors.OutOfRangeError:
                epoch += 1
                session.run(iterator.initializer)
                break


if __name__ == '__main__':
    tf.app.run()
