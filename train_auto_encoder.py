# -*- coding: utf-8 -*-
from model import AE

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Number of epoch")
flags.DEFINE_integer("batch_size", 16, "The size of batch images")
flags.DEFINE_integer("image_size", 256, "The size of image to use")
flags.DEFINE_integer("label_size", 252, "The size of label to produce")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color")
flags.DEFINE_integer("stride", 50, "The size of stride to apply input image")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_string("summary_dir", "log", "Name of log directory")
flags.DEFINE_boolean("is_train", True, "True for training")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session() as sess:
        ae= AE(sess, 
                     image_size=FLAGS.image_size, 
                     label_size=FLAGS.label_size, 
                     batch_size=FLAGS.batch_size,
                     c_dim=FLAGS.c_dim, 
                     checkpoint_dir=FLAGS.checkpoint_dir)

        ae.train(FLAGS)
    
if __name__ == '__main__':
    tf.app.run()
