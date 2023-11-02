# -*- coding: utf-8 -*-
from utils import (
    read_data, 
    auto_encoder_input_setup, 
    gradient
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class AE(object):

    def __init__(self, 
                 sess, 
                 image_size=256,
                 label_size=252,
                 batch_size=16,
                 c_dim=1, 
                 checkpoint_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        
    def build_model(self):
        with tf.name_scope('COCO_input1'):
            self.images_1 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_1')
            self.labels_1 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_1')
        with tf.name_scope('COCO_input2'):
            self.images_2 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_2')
            self.labels_2 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_2')
        
        with tf.name_scope('input'):
            self.input_image=tf.concat([self.images_1,self.images_2],axis=0)
            self.input_label=tf.concat([self.labels_1,self.labels_2],axis=0)
        
        with tf.name_scope('auto_encoder'): 
            self.output_image=self.auto_encoder_model(self.input_image)
        
        with tf.name_scope('a_loss'):
            self.a_loss_1=tf.reduce_mean(tf.square(self.output_image - self.input_label))
            self.a_loss_2=tf.reduce_mean(tf.square(gradient(self.output_image) - gradient (self.input_label)))
            self.a_loss_total=5.0*self.a_loss_1+ 1.0*self.a_loss_2
        self.saver = tf.train.Saver(max_to_keep=50)
        
    def train(self, config):
        auto_encoder_input_setup(config,"Train_images/Train_coco1", "Train_AE/coco1.h5")
        auto_encoder_input_setup(config,"Train_images/Train_coco2", "Train_AE/coco2.h5")
        data_dir_1 = os.path.join('./{}'.format(config.checkpoint_dir), "Train_AE","coco1.h5")
        data_dir_2 = os.path.join('./{}'.format(config.checkpoint_dir), "Train_AE","coco2.h5")
        train_data_1, train_label_1 = read_data(data_dir_1)
        train_data_2, train_label_2 = read_data(data_dir_2)
        t_vars = tf.trainable_variables()
        self.a_vars = [var for var in t_vars if 'decoder_model' or 'encoder_model' in var.name]
        print(self.a_vars)

        with tf.name_scope('train_step'):
            self.train_auto_encoder_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.a_loss_total,var_list=self.a_vars)
        
        tf.summary.scalar('loss_total', self.a_loss_total)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
        tf.initialize_all_variables().run()
        counter = 0
        start_time = time.time()

        if config.is_train:
            print("Training...")

            for ep in range(config.epoch):
                # Run by batch images
                # print(ep)
                batch_idxs = len(train_data_1) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images_1 = train_data_1[idx*config.batch_size : (idx+1)*config.batch_size]
                    batch_labels_1 = train_label_1[idx*config.batch_size : (idx+1)*config.batch_size]
                    batch_images_2 = train_data_2[idx*config.batch_size : (idx+1)*config.batch_size]
                    batch_labels_2 = train_label_2[idx*config.batch_size : (idx+1)*config.batch_size]

                    counter += 1
    
                    _, err_a,summary_str= self.sess.run([self.train_auto_encoder_op, self.a_loss_total,self.summary_op], feed_dict=
                                                            {self.images_1: batch_images_1, self.images_2: batch_images_2, self.labels_1: batch_labels_1,self.labels_2:batch_labels_2})
                    
                    self.train_writer.add_summary(summary_str,counter)

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_a: [%.8f]" \
                              % ((ep+1), counter, time.time()-start_time, err_a))

                self.save(config.checkpoint_dir, ep)        
    
    def auto_encoder_model(self,img):
        with tf.variable_scope('encoder_model'):
            with tf.variable_scope('layer_1'):
                conv1 = tf.layers.conv2d(img,16,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_2'):
                conv2 = tf.layers.conv2d(conv1,32,(3,3),(2,2), activation='relu', padding='same')
            with tf.variable_scope('layer_3'):
                conv3 = tf.layers.conv2d(conv2,64,(3,3),(2,2), activation='relu', padding='same')
            with tf.variable_scope('layer_4'):
                conv4 = tf.layers.conv2d(conv3,128,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_5'):
                conv5 = tf.layers.conv2d(conv4,128,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_6'):
                conv6 = tf.layers.conv2d(conv5,128,(3,3), activation='relu', padding='same')
                
        with tf.variable_scope('decoder_model'):        
            with tf.variable_scope('layer_7'):
                conv7 = tf.layers.conv2d(conv6,128,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_8'):
                conv8 = tf.layers.conv2d(conv7,128,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_9'):
                shape = conv2.get_shape().as_list()
                out_shape = [shape[1], shape[2]]
                upsample9 = tf.image.resize_bilinear(conv8, out_shape)
                dconv9 = tf.layers.conv2d(upsample9,64,(3,3), activation='relu', padding='same')
            with tf.variable_scope('layer_10'):
                shape = conv1.get_shape().as_list()
                out_shape = [shape[1], shape[2]]
                upsample10 = tf.image.resize_bilinear(dconv9, out_shape)
                dconv10 = tf.layers.conv2d(upsample10,32,(3,3), activation='relu', padding='valid')
            with tf.variable_scope('layer_11'):
                conv11 = tf.layers.conv2d(dconv10,32,(3,3), activation='relu', padding='valid')
            with tf.variable_scope('layer_12'):
                conv12 = tf.layers.conv2d(conv11,1,(1,1), activation='tanh', padding='valid')
        return conv12
    
    def save(self, checkpoint_dir, step):
        model_name = "AE.model"
        model_dir = "%s_%s" % ("AE", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("AE", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
            return True
        else:
            return False