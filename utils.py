# -*- coding: utf-8 -*-
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
import cv2


def read_data(path):
    """
    Read h5 format data file
  
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def prepare_data(dataset):
    """
    Retrieves the path of all images from the directory and sorts them
    
    Args:
        dataset: a folder
    """
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x:int(x[len(data_dir)+16:-4]))
    return data


def make_data(data, label, h5dir):
    """
    Make input data and generate h5 file
    
    """
    savepath = os.path.join('.', os.path.join('checkpoint', h5dir))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint', os.path.dirname(h5dir)))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint', os.path.dirname(h5dir))))

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
    

def imread(path, is_grayscale=True):
    """
    Read image using its path, YCbCr format
    Default value is gray-scale(1 channel)
    
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)



def auto_encoder_input_setup(config, data_dir, h5dir, mode='resize'):
    """
    Read image files from data_dir, make their sub-images and saved them as .h5 file.
    
    """

    data = prepare_data(data_dir)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = int(abs(config.image_size - config.label_size) / 2)

    if mode == 'crop':
        for i in range(len(data)):
            input_=(imread(data[i])-127.5)/127.5
            label_=input_
    
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
                
            for x in range(0, h-config.image_size+1, config.stride):
                for y in range(0, w-config.image_size+1, config.stride):
                    sub_input = input_[x:x+config.image_size, y:y+config.image_size]
                    
                    sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size]
                    
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
                    
                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    elif mode == 'resize':
        for i in range(len(data)):
            input_=(imread(data[i])-127.5)/127.5
            # resize
            input_ = np.expand_dims(input_, axis=-1)
            sub_input_ = cv2.resize(input_, dsize=(config.image_size, config.image_size), interpolation=cv2.INTER_CUBIC)
            sub_input_ = sub_input_.reshape([config.image_size,config.image_size,1])
            sub_label_ = sub_input_[padding:-padding,padding:-padding]

            sub_input_sequence.append(sub_input_)
            sub_label_sequence.append(sub_label_)
    
    # Convert list to numpy array
    arrdata = np.asarray(sub_input_sequence) 
    arrlabel = np.asarray(sub_label_sequence) 
    print(arrdata.shape)
    print(arrlabel.shape)
    make_data(arrdata, arrlabel, h5dir)


def gradient(input_image):
    """
    Laplace gradient image
    
    """
    filter=tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[3,3,1,1])
    d=tf.nn.conv2d(input_image,filter,strides=[1,1,1,1], padding='SAME')
    return d
