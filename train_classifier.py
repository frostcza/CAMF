# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import cv2

def classifier_input_setup(data_dir='Train_CL'):
    data_ir = prepare_data(dataset="Train_images/Train_ir_tno")
    data_vi = prepare_data(dataset="Train_images/Train_vi_tno")

    sub_input_sequence = []
    sub_label_sequence = []

    for i in range(len(data_ir)):
        input_ir=(imread(data_ir[i])-127.5)/127.5
        input_vi=(imread(data_vi[i])-127.5)/127.5
        
        # crop
        if len(input_ir.shape) == 3:
            h, w, _ = input_ir.shape
        else:
            h, w = input_ir.shape
        
        for x in range(0, h-image_size+1, stride):
            for y in range(0, w-image_size+1, stride):
                sub_input_ir = input_ir[x:x+image_size, y:y+image_size]
                sub_label_ir = [1,0]
                sub_input_ir = sub_input_ir.reshape([image_size,image_size, 1]) 
                
                sub_input_sequence.append(sub_input_ir)
                sub_label_sequence.append(sub_label_ir)
                
                sub_input_vi = input_vi[x:x+image_size, y:y+image_size]
                sub_label_vi = [0,1]
                sub_input_vi = sub_input_vi.reshape([image_size,image_size, 1])
                
                sub_input_sequence.append(sub_input_vi)
                sub_label_sequence.append(sub_label_vi)


    arrdata = np.asarray(sub_input_sequence) 
    arrlabel = np.asarray(sub_label_sequence) 
    print(arrdata.shape)
    print(arrlabel.shape)
    make_data(arrdata, arrlabel,data_dir)
    
    
def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def prepare_data(dataset):
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data


def make_data(data, label, data_dir):
    savepath = os.path.join('.', os.path.join('checkpoint',data_dir,'tno.h5'))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint',data_dir))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint',data_dir)))

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
    

def imread(path, is_grayscale=True):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def save(sess, saver, checkpoint_dir, step, kf):
    model_name = "CL.model"
    model_dir = "%s_%s_kf%s" % ("CL", str(2), str(kf))
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)
        

def classifier(img):
    reader_ae = tf.train.NewCheckpointReader('./checkpoint/AE/AE.model-1')
    with tf.variable_scope('encoder_model',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_1/conv2d/kernel')))
            bias=tf.get_variable("b_1",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_1/conv2d/bias')))
            conv1= tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1 = tf.nn.relu(conv1)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_2/conv2d/kernel')))
            bias=tf.get_variable("b_2",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_2/conv2d/bias')))
            conv2= tf.nn.conv2d(conv1, weights, strides=[1,2,2,1], padding='SAME') + bias
            conv2 = tf.nn.relu(conv2)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_3/conv2d/kernel')))
            bias=tf.get_variable("b_3",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_3/conv2d/bias')))
            conv3= tf.nn.conv2d(conv2, weights, strides=[1,2,2,1], padding='SAME') + bias
            conv3 = tf.nn.relu(conv3)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_4/conv2d/kernel')))
            bias=tf.get_variable("b_4",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_4/conv2d/bias')))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.relu(conv4)
        with tf.variable_scope('layer_5'):
            weights=tf.get_variable("w_5",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_5/conv2d/kernel')))
            bias=tf.get_variable("b_5",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_5/conv2d/bias')))
            conv5= tf.nn.conv2d(conv4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5 = tf.nn.relu(conv5)
        with tf.variable_scope('layer_6'):
            weights=tf.get_variable("w_6",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_6/conv2d/kernel')))
            bias=tf.get_variable("b_6",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_6/conv2d/bias')))
            conv6= tf.nn.conv2d(conv5, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6 = tf.nn.relu(conv6)
        
    with tf.variable_scope('classifier_model'):
        with tf.variable_scope('layer_7'):
            weights=tf.get_variable("w_7_1",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_7_1",[128],initializer=tf.constant_initializer(0.0))
            conv7_1=tf.nn.depthwise_conv2d(conv6, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv7_1=tf.nn.relu(conv7_1)
            
            weights=tf.get_variable("w_7_2",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_7_2",[128],initializer=tf.constant_initializer(0.0))
            conv7_2=tf.nn.depthwise_conv2d(conv6, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv7_2=tf.nn.relu(conv7_2)
            
            weights=tf.get_variable("w_7_3",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_7_3",[128],initializer=tf.constant_initializer(0.0))
            conv7_3=tf.nn.depthwise_conv2d(conv6, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv7_3=tf.nn.relu(conv7_3)
            
            weights=tf.get_variable("w_7_4",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_7_4",[128],initializer=tf.constant_initializer(0.0))
            conv7_4=tf.nn.depthwise_conv2d(conv6, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv7_4=tf.nn.relu(conv7_4)
            
            conv7 = conv7_1 + conv7_2 + conv7_3 + conv7_4
            
        with tf.variable_scope('layer_8'):
            weights=tf.get_variable("w_8_1",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_8_1",[128],initializer=tf.constant_initializer(0.0))
            conv8_1=tf.nn.depthwise_conv2d(conv7, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv8_1=tf.nn.relu(conv8_1)
            
            weights=tf.get_variable("w_8_2",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_8_2",[128],initializer=tf.constant_initializer(0.0))
            conv8_2=tf.nn.depthwise_conv2d(conv7, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv8_2=tf.nn.relu(conv8_2)
            
            weights=tf.get_variable("w_8_3",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_8_3",[128],initializer=tf.constant_initializer(0.0))
            conv8_3=tf.nn.depthwise_conv2d(conv7, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv8_3=tf.nn.relu(conv8_3)
            
            weights=tf.get_variable("w_8_4",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_8_4",[128],initializer=tf.constant_initializer(0.0))
            conv8_4=tf.nn.depthwise_conv2d(conv7, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv8_4=tf.nn.relu(conv8_4)
            
            conv8 = conv8_1 + conv8_2 + conv8_3 + conv8_4

            
        with tf.variable_scope('layer_9'):
            weights=tf.get_variable("w_9_1",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_9_1",[128],initializer=tf.constant_initializer(0.0))
            conv9_1=tf.nn.depthwise_conv2d(conv8, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv9_1=tf.nn.relu(conv9_1)
            
            weights=tf.get_variable("w_9_2",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_9_2",[128],initializer=tf.constant_initializer(0.0))
            conv9_2=tf.nn.depthwise_conv2d(conv8, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv9_2=tf.nn.relu(conv9_2)
            
            weights=tf.get_variable("w_9_3",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_9_3",[128],initializer=tf.constant_initializer(0.0))
            conv9_3=tf.nn.depthwise_conv2d(conv8, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv9_3=tf.nn.relu(conv9_3)
            
            weights=tf.get_variable("w_9_4",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_9_4",[128],initializer=tf.constant_initializer(0.0))
            conv9_4=tf.nn.depthwise_conv2d(conv8, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv9_4=tf.nn.relu(conv9_4)

            conv9 = conv9_1 + conv9_2 + conv9_3 + conv9_4
            
        with tf.variable_scope('layer_10'):
            weights=tf.get_variable("w_10_1",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_10_1",[128],initializer=tf.constant_initializer(0.0))
            conv10_1=tf.nn.depthwise_conv2d(conv9, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv10_1=tf.nn.relu(conv10_1)
            
            weights=tf.get_variable("w_10_2",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_10_2",[128],initializer=tf.constant_initializer(0.0))
            conv10_2=tf.nn.depthwise_conv2d(conv9, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv10_2=tf.nn.relu(conv10_2)
            
            weights=tf.get_variable("w_10_3",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_10_3",[128],initializer=tf.constant_initializer(0.0))
            conv10_3=tf.nn.depthwise_conv2d(conv9, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv10_3=tf.nn.relu(conv10_3)
            
            weights=tf.get_variable("w_10_4",[3,3,128,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("b_10_4",[128],initializer=tf.constant_initializer(0.0))
            conv10_4=tf.nn.depthwise_conv2d(conv9, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv10_4=tf.nn.relu(conv10_4)

            conv10 = conv10_1 + conv10_2 + conv10_3 + conv10_4
        
        with tf.variable_scope('layer_12'):
            gap =tf.reduce_mean(conv10, axis=[1, 2])
            flat=tf.layers.flatten(gap)
            dense = tf.layers.dense(inputs=flat, units=2)
            # dense = tf.layers.dense(inputs=flat, units=2, use_bias=False)
            
    return dense

beginTime = time.time()

image_size = 256
stride = 24
batch_size = 16
learning_rate = 0.0001
max_steps = 20000
k = 5

# placeholders
images = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 1])
labels = tf.placeholder(tf.float32, shape=[None, 2])

# model
class_output=classifier(images)

# loss function
c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_output,labels=labels))
writer = tf.summary.FileWriter('log/cl', tf.get_default_graph())
tf.summary.scalar('loss', c_loss)           
merge_summary = tf.summary.merge_all()  

# comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(class_output, 1), tf.argmax(labels,1))

# calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=200)

with tf.Session() as sess:
    # classifier_input_setup()
    data_dir = os.path.join('./{}'.format("checkpoint"), "Train_CL","tno.h5")
    all_data, all_label = read_data(data_dir)
    
    random.seed(0)
    random.shuffle(all_data)
    random.seed(0)
    random.shuffle(all_label)
    acc_list = []
    for kf in range(0,k):
        print('Fold: %s'%(str(kf)))
        part_len = int(np.floor(len(all_data) / k))
        index = np.array(range(0, len(all_data)))
        dev_index = index[kf*part_len:(kf+1)*part_len]
        train_index = np.concatenate((index[0:kf*part_len],index[(kf+1)*part_len:-1]),axis=-1)
        
        train_data = all_data[train_index]
        train_label = all_label[train_index]
        dev_data = all_data[dev_index]
        dev_label = all_label[dev_index]
        
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'classifier_model' in var.name]
        # print(c_vars)
        
        # training
        train_classifier_op = tf.train.RMSPropOptimizer(learning_rate).minimize(c_loss,var_list=c_vars)
        
        sess.run(tf.global_variables_initializer())
        
        # Repeat max_steps times
        for i in range(max_steps):
    
            # Generate input data batch
            indices = np.random.choice(train_data.shape[0], batch_size)
            images_batch = train_data[indices]
            labels_batch = train_label[indices]
    
            # Periodically print out the model's current accuracy
            if i % 100 == 0:
                test_accuracy = 0
                train_loss = sess.run(c_loss, feed_dict={images: images_batch, labels: labels_batch})
                print('kf {:5d} Step {:5d}: training loss {:g}'.format(kf, i, train_loss))
                save(sess, saver, 'checkpoint', int(i/1000), kf)
                
                # evaluate on the dev set
                num_batch = int(np.floor(dev_data.shape[0] / batch_size))
                for ii in range(num_batch):
                    images_batch = dev_data[ii*batch_size:(ii+1)*batch_size]
                    labels_batch = dev_label[ii*batch_size:(ii+1)*batch_size]
                    test_accuracy += sess.run(accuracy, feed_dict={images: images_batch, labels: labels_batch})
                print('validation accuracy {:g}'.format(test_accuracy/num_batch))
                
            # Perform a single training step
            _,train_summary = sess.run([train_classifier_op,merge_summary], feed_dict={images: images_batch,labels: labels_batch})
            writer.add_summary(train_summary,i)
        acc_list.append(test_accuracy/num_batch)
    print("accuracy list: ")
    print(acc_list)
    print('{:1d} Fold average accuracy {:g}'.format(k, np.mean(np.array(acc_list))))

writer.close()
endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
