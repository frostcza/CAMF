# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import copy

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)


def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


def imread(path, is_grayscale=True):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
    return scipy.misc.imsave(path, image)

def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def encoder_model(img):
    with tf.variable_scope('encoder_model',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_1/conv2d/kernel')))
            bias=tf.get_variable("b_1",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_1/conv2d/bias')))
            conv1_c= tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_c = tf.nn.relu(conv1_c)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_2/conv2d/kernel')))
            bias=tf.get_variable("b_2",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_2/conv2d/bias')))
            conv2_c= tf.nn.conv2d(conv1_c, weights, strides=[1,2,2,1], padding='SAME') + bias
            conv2_c = tf.nn.relu(conv2_c)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_3/conv2d/kernel')))
            bias=tf.get_variable("b_3",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_3/conv2d/bias')))
            conv3_c= tf.nn.conv2d(conv2_c, weights, strides=[1,2,2,1], padding='SAME') + bias
            conv3_c = tf.nn.relu(conv3_c)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_4/conv2d/kernel')))
            bias=tf.get_variable("b_4",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_4/conv2d/bias')))
            conv4_c= tf.nn.conv2d(conv3_c, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_c = tf.nn.relu(conv4_c)
        with tf.variable_scope('layer_5'):
            weights=tf.get_variable("w_5",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_5/conv2d/kernel')))
            bias=tf.get_variable("b_5",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_5/conv2d/bias')))
            conv5_c= tf.nn.conv2d(conv4_c, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_c = tf.nn.relu(conv5_c)
        with tf.variable_scope('layer_6'):
            weights=tf.get_variable("w_6",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_6/conv2d/kernel')))
            bias=tf.get_variable("b_6",initializer=tf.constant(reader_ae.get_tensor('encoder_model/layer_6/conv2d/bias')))
            conv6_c= tf.nn.conv2d(conv5_c, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_c = tf.nn.relu(conv6_c)

        weights_line=tf.get_variable("w_12",initializer=tf.constant(reader_cl.get_tensor('classifier_model/layer_12/dense/kernel')))
        return conv6_c, weights_line, tf.shape(conv2_c), tf.shape(conv1_c)


def decoder_model(fused_feature,size2,size1):
    with tf.variable_scope('decoder_model',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_7'):
            weights=tf.get_variable("w_7",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_7/conv2d/kernel')))
            bias=tf.get_variable("b_7",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_7/conv2d/bias')))
            conv7_d= tf.nn.conv2d(fused_feature, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_d = tf.nn.relu(conv7_d)
        with tf.variable_scope('layer_8'):
            weights=tf.get_variable("w_8",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_8/conv2d/kernel')))
            bias=tf.get_variable("b_8",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_8/conv2d/bias')))
            conv8_d= tf.nn.conv2d(conv7_d, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_d = tf.nn.relu(conv8_d)

        with tf.variable_scope('layer_9'):
            out_shape = [size2[1], size2[2]]
            upsample9_d = tf.image.resize_bilinear(conv8_d, out_shape)
            weights=tf.get_variable("w_9",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_9/conv2d/kernel')))
            bias=tf.get_variable("b_9",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_9/conv2d/bias')))
            dconv9_d= tf.nn.conv2d(upsample9_d, weights, strides=[1,1,1,1], padding='SAME') + bias
            dconv9_d = tf.nn.relu(dconv9_d)

        with tf.variable_scope('layer_10'):
            out_shape = [size1[1], size1[2]]
            upsample10_d = tf.image.resize_bilinear(dconv9_d, out_shape)
            weights=tf.get_variable("w_10",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_10/conv2d/kernel')))
            bias=tf.get_variable("b_10",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_10/conv2d/bias')))
            dconv10_d= tf.nn.conv2d(upsample10_d, weights, strides=[1,1,1,1], padding='VALID') + bias
            dconv10_d = tf.nn.relu(dconv10_d)


        with tf.variable_scope('layer_11'):
            weights=tf.get_variable("w_11",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_11/conv2d/kernel')))
            bias=tf.get_variable("b_11",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_11/conv2d/bias')))
            conv11_d= tf.nn.conv2d(dconv10_d, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv11_d = tf.nn.relu(conv11_d)

        with tf.variable_scope('layer_12'):
            weights=tf.get_variable("w_12",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_12/conv2d/kernel')))
            bias=tf.get_variable("b_12",initializer=tf.constant(reader_ae.get_tensor('decoder_model/layer_12/conv2d/bias')))
            conv12_d= tf.nn.conv2d(conv11_d, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv12_d = tf.nn.tanh(conv12_d)

    return conv12_d


def fusion_model(feature_ir,feature_vi,weight_ir,weight_vi):
    # CAMF
    w_ir = 0.5*tf.nn.tanh(weight_ir) + 0.5
    w_vi = 0.5*tf.nn.tanh(weight_vi) + 0.5
    if test_dataset == 'tno' or test_dataset == 'medical':
        w_ir = w_ir/tf.reduce_mean(w_ir, axis=-1)*1.0
        w_vi = w_vi/tf.reduce_mean(w_vi, axis=-1)*1.0
    elif test_dataset == 'roadscene':
        w_ir = w_ir/tf.reduce_mean(w_ir, axis=-1)*0.7
        w_vi = w_vi/tf.reduce_mean(w_vi, axis=-1)*0.7
    fused_feature = tf.multiply(feature_ir, w_ir) + tf.multiply(feature_vi, w_vi)

    # # mean
    # fused_feature = 0.5*feature_vi + 0.5*feature_ir

    # # add
    # fused_feature = feature_vi + feature_ir

    # # max
    # fused_feature = tf.maximum(feature_vi,feature_ir)

    # # l1_norm
    # abs_ir = tf.abs(feature_ir)
    # abs_vi = tf.abs(feature_vi)
    # l1_ir = tf.reduce_sum(abs_ir,3)
    # l1_vi = tf.reduce_sum(abs_vi,3)
    # mask_ir = tf.reduce_sum(l1_ir,0)
    # mask_vi = tf.reduce_sum(l1_vi,0)
    # m_ir = mask_ir/(mask_ir+mask_vi)
    # m_vi = mask_vi/(mask_ir+mask_vi)
    # feature_ir = tf.squeeze(feature_ir,0)
    # feature_vi = tf.squeeze(feature_vi,0)
    # fused_feature = tf.multiply(m_ir,feature_ir[:,:,0]) + tf.multiply(m_vi,feature_vi[:,:,0])
    # fused_feature = tf.expand_dims(fused_feature,-1)
    # for i in range(1,128):
    #     add_feature = tf.multiply(m_ir,feature_ir[:,:,i]) + tf.multiply(m_vi,feature_vi[:,:,i])
    #     add_feature = tf.expand_dims(add_feature,-1)
    #     fused_feature = tf.concat([fused_feature,add_feature],-1)
    # fused_feature = tf.expand_dims(fused_feature,0)

    return fused_feature

def input_setup(index):
    padding=2
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    test_data_ir= np.asarray(sub_ir_sequence)
    test_data_vi= np.asarray(sub_vi_sequence)
    return test_data_ir,test_data_vi

'''--------------------------------------------------------------------------------------'''
# test_dataset = 'tno'
# test_dataset = 'roadscene'
# test_dataset = 'medical'

if test_dataset == 'tno':
    reader_ae = tf.train.NewCheckpointReader('./checkpoint/AE/AE.model-1')
    reader_cl = tf.train.NewCheckpointReader('./checkpoint/CL/CL.model-tno')
elif test_dataset == 'roadscene':
    reader_ae = tf.train.NewCheckpointReader('./checkpoint/AE/AE.model-2')
    reader_cl = tf.train.NewCheckpointReader('./checkpoint/CL/CL.model-roadscene')
elif test_dataset == 'medical':
    reader_ae = tf.train.NewCheckpointReader('./checkpoint/AE/AE.model-1')
    reader_cl = tf.train.NewCheckpointReader('./checkpoint/CL/CL.model-medical')

with tf.name_scope('IR_input'):
    images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
with tf.name_scope('VI_input'):
    images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
with tf.name_scope('encoder'):
    feature_ir, weights_line, size2, size1 = encoder_model(images_ir)
    weight_vi = weights_line[:,1]
    weight_ir = weights_line[:,0]
    feature_vi, _, _, _ = encoder_model(images_vi)
    fused_feature = fusion_model(feature_ir,feature_vi,weight_ir,weight_vi)
    output_img = decoder_model(fused_feature,size2,size1)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    if test_dataset == 'tno':
        data_ir=prepare_data('Test_images/tno/IR')
        data_vi=prepare_data('Test_images/tno/VI')
    elif test_dataset == 'roadscene':
        data_ir=prepare_data('Test_images/roadscene/IR')
        data_vi=prepare_data('Test_images/roadscene/VI')
    elif test_dataset == 'medical':
        data_ir=prepare_data('Test_images/medical/PET')
        data_vi=prepare_data('Test_images/medical/MRI')

    time_list = []

    for i in range(len(data_ir)):
        start=time.time()
        test_data_ir,test_data_vi=input_setup(i)
        result =sess.run(output_img,feed_dict={images_ir: test_data_ir,images_vi: test_data_vi})
        if test_dataset == 'tno':
            pathname = 'Fused_tno'
        elif test_dataset == 'roadscene':
            pathname = 'Fused_roadscene'
        elif test_dataset == 'medical':
            pathname = 'Fused_medical'


        if test_dataset == 'medical':
            result = result.squeeze()
            result=result*127.5+127.5
            ori_input = imread(data_ir[i], False)
            ori_input[:,:,0] = result
            result = ycbcr2rgb(ori_input)
            image_path = os.path.join(os.getcwd(), 'result',pathname)
            name = os.path.basename(data_ir[i])
            name = name.split('.')
            name = name[0]
            image_path = os.path.join(image_path, name+".bmp")
            imsave(result, image_path)
        else:
            result=result*127.5+127.5
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), 'result',pathname)
            name = os.path.basename(data_ir[i])
            name = name.split('.')
            name = name[0]
            image_path = os.path.join(image_path, name+".bmp")
            imsave(result, image_path)

        end=time.time()
        time_list.append(end-start)
        print("Testing [%d] success,Testing time is [%f]"%(i,end-start))

    del time_list[0]
    print("Average testing time is [%f]"%(np.mean(time_list)))
    print("Min testing time is [%f]"%(np.min(time_list)))
    print("Max testing time is [%f]"%(np.max(time_list)))
tf.reset_default_graph()
