import cv2
import numpy
from tensorflow import tanh
import tensorflow.contrib.slim as slim

from loss_util import gdl_loss, mse_loss, ssim_loss
from loss_util import ssim_loss
from utils import (
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *
import vgg

import drcan_metric
import drcan_tfutil as tfu


import tensorflow.contrib.layers as tcl

###  pix2pix + multiply + add + enhance
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=230,
               image_width=310,
               label_height=230, 
               label_width=310,
               batch_size=16,
               c_dim=3, 
               checkpoint_dir=None, 
               sample_dir=None

               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5


    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.vgg_dir='./vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    # self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    # self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    # self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    # self.images_eh = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_eh')
    #self.images_sm = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_sm')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')


    self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    # self.images_test_wb = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_wb')
    # self.images_test_ce = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_ce')
    # self.images_test_gc = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_gc')
    # self.images_test_eh = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],name='images_test_eh')
    #self.images_test_sm = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],name='images_test_sm')
    self.labels_test = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test')
    
    self.pred_h1= self.model()

    self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.loss_texture1 =tf.reduce_mean(tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))

    # ##  添加 image gradient difference--------2020 09 08
    # self.gradient = loss_gradient_difference(self.labels_image, self.pred_h1)
    # ##
    # self.loss_h1= tf.reduce_mean(tf.abs(self.labels_image-self.pred_h1))

    # edge loss
    # self.edge_loss = tf.reduce_mean(tf.squared_difference(tf.image.sobel_edges(self.labels_image), tf.image.sobel_edges(self.pred_h1)))
    # self.loss = 0.05*self.loss_texture1+ self.loss_h1  + 0.01 * self.edge_loss
    self.ssim_loss = ssim_loss(self.labels_image, self.pred_h1)
    self.gdl_loss = gdl_loss(self.labels_image, self.pred_h1)
    self.mse_loss = mse_loss(self.labels_image, self.pred_h1)
    self.loss = 0.05 * self.loss_texture1 + self.mse_loss + self.ssim_loss + 0.01 * self.gdl_loss### final
    # self.loss = 0.05 * self.loss_texture1 + self.mse_loss + self.ssim_loss ###  -gdl loss
    # self.loss = self.mse_loss + self.ssim_loss + 0.01 * self.gdl_loss###  -vgg loss
    # self.loss = 0.05 * self.loss_texture1 + self.ssim_loss + 0.01 * self.gdl_loss### sub mse
    # self.loss = 0.05 * self.loss_texture1 + self.mse_loss  + 0.01 * self.gdl_loss### sub ssim loss
    # self.loss = 0.05 * self.loss_texture1## only vgg loss

    t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver(max_to_keep=0)
    
  def train(self, config):
    if config.is_train:     
      data_train_list = prepare_data(self.sess, dataset=r"F:\chenlong\UIE_MRJL_ChinaMM2021\train_and_test_dataset_uieb\input_train")
      # data_wb_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_wb_train")
      # data_ce_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_ce_train")
      # data_gc_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_gc_train")
      # data_eh_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_eh_train")
      #data_sm_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_sm_train")
      image_train_list = prepare_data(self.sess, dataset=r"F:\chenlong\UIE_MRJL_ChinaMM2021/train_and_test_dataset_uieb/gt_train")

      data_test_list = prepare_data(self.sess, dataset=r"F:\chenlong\UIE_MRJL_ChinaMM2021/train_and_test_dataset_uieb/input_test_")
      # data_wb_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_wb_test")
      # data_ce_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_ce_test")
      # data_gc_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_gc_test")
      # data_eh_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_eh_test")
      #data_sm_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_sm_test")
      image_test_list = prepare_data(self.sess, dataset=r"F:\chenlong\UIE_MRJL_ChinaMM2021/train_and_test_dataset_uieb/gt_test")

      seed = 568
      np.random.seed(seed)
      np.random.shuffle(data_train_list)
      # np.random.seed(seed)
      # np.random.shuffle(data_wb_train_list)
      # np.random.seed(seed)
      # np.random.shuffle(data_ce_train_list)
      # np.random.seed(seed)
      # np.random.shuffle(data_gc_train_list)
      # np.random.seed(seed)
      # np.random.shuffle(data_eh_train_list)
      #np.random.seed(seed)
      #np.random.shuffle(data_sm_train_list)
      np.random.seed(seed)
      np.random.shuffle(image_train_list)

    else:
      data_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/input_test_200")
      # data_wb_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_wb_test")
      # data_ce_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_ce_test")
      # data_gc_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_gc_test")
      # data_eh_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_eh_test")
      #data_sm_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_uieb/input_sm_test")
      image_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/gt_test_200")



    sample_data_files = data_test_list[16:20]
    # sample_wb_data_files = data_wb_test_list[16:20]
    # sample_ce_data_files = data_ce_test_list[16:20]
    # sample_gc_data_files = data_gc_test_list[16:20]
    # sample_eh_data_files = data_eh_test_list[16:20]
    #sample_sm_data_files = data_sm_test_list[16:20]
    sample_image_files = image_test_list[16:20]

    sample_data = [
          get_image(sample_data_file,
                    is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
    sample_lable_image = [
          get_image(sample_image_file,
                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

    sample_inputs_data = np.array(sample_data).astype(np.float32)
    sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


    self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss = np.ones(config.epoch)



      for ep in range(config.epoch):
        # Run by batch images
        learning_rate = 0.00005
        if ep < 200:
            config.learning_rate = learning_rate
        elif ep>=200 and ep <400:
            config.learning_rate = learning_rate/5
        elif ep>=400 and ep<=500:
            config.learning_rate = 0.00008
        print("learning_rate:",config.learning_rate)
        print("\n")
        
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):

          batch_files          = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          # batch_files_wb       = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          # batch_files_ce       = data_ce_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          # batch_files_gc       = data_gc_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          # batch_files_eh       = data_eh_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
          #batch_files_sm       = data_sm_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_image_files    = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]


          batch_ = [
          get_image(batch_file,
                    is_grayscale=self.is_grayscale) for batch_file in batch_files]
          # batch_wb = [
          # get_image(batch_wb_file,
          #           is_grayscale=self.is_grayscale) for batch_wb_file in batch_files_wb]
          # batch_ce = [
          # get_image(batch_ce_file,
          #           is_grayscale=self.is_grayscale) for batch_ce_file in batch_files_ce]
          # batch_gc = [
          # get_image(batch_gc_file,
          #           is_grayscale=self.is_grayscale) for batch_gc_file in batch_files_gc]
          # batch_eh = [
          #     get_image(batch_eh_file,
          #               is_grayscale=self.is_grayscale) for batch_eh_file in batch_files_eh]
          # batch_sm = [
          #     get_image(batch_sm_file,
          #               is_grayscale=self.is_grayscale) for batch_sm_file in batch_files_sm]
          batch_labels_image = [
          get_image(batch_image_file,
                    is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]

          
          batch_input = np.array(batch_).astype(np.float32)
          # batch_wb_input = np.array(batch_wb).astype(np.float32)
          # batch_ce_input = np.array(batch_ce).astype(np.float32)
          # batch_gc_input = np.array(batch_gc).astype(np.float32)
          # batch_eh_input = np.array(batch_eh).astype(np.float32)
          #batch_sm_input = np.array(batch_sm).astype(np.float32)
          batch_image_input = np.array(batch_labels_image).astype(np.float32)

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss ], feed_dict={self.images: batch_input, self.labels_image:batch_image_input})
          # print(batch_light)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err ))
            
          if idx  == batch_idxs-1: 
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_train_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              # sample_wb_files = data_wb_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              # sample_ce_files = data_ce_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              # sample_gc_files = data_gc_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              # sample_eh_files = data_eh_train_list[idx_test * config.batch_size: (idx_test + 1) * config.batch_size]
              #sample_sm_files = data_sm_train_list[idx_test * config.batch_size: (idx_test + 1) * config.batch_size]
              sample_image_files = image_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
             
              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
              # sample_wb_image = [get_image(sample_wb_file,
              #                       is_grayscale=self.is_grayscale) for sample_wb_file in sample_wb_files]
              # sample_ce_image = [get_image(sample_ce_file,
              #                       is_gmmodelrayscale=self.is_grayscale) for sample_ce_file in sample_ce_files]
              # sample_gc_image = [get_image(sample_gc_file,
              #                       is_grayscale=self.is_grayscale) for sample_gc_file in sample_gc_files]
              # sample_eh_image = [get_image(sample_eh_file,
              #                              is_grayscale=self.is_grayscale) for sample_eh_file in sample_eh_files]
              # sample_sm_image = [get_image(sample_sm_file,
              #                              is_grayscale=self.is_grayscale) for sample_sm_file in sample_sm_files]

              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

              sample_inputs_data = np.array(sample_data).astype(np.float32)
              # sample_inputs_wb_image = np.array(sample_wb_image).astype(np.float32)
              # sample_inputs_ce_image = np.array(sample_ce_image).astype(np.float32)
              # sample_inputs_gc_image = np.array(sample_gc_image).astype(np.float32)
              # sample_inputs_eh_image = np.array(sample_eh_image).astype(np.float32)
              #sample_inputs_sm_image = np.array(sample_sm_image).astype(np.float32)
              sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data,self.labels_image:sample_inputs_lable_image})

            loss[ep]=np.mean(err_test)
            print(loss)
            self.save(config.checkpoint_dir, counter)

  def gaussian(self,ori_image, down_times=5):
    # 1：添加第一个图像为原始图像
      temp_gau = ori_image.copy()
      gaussian_pyramid = [temp_gau]
      for i in range(down_times):
        # 2：连续存储5次下采样，这样高斯金字塔就有6层
          temp_gau = cv2.pyrDown(temp_gau)
          gaussian_pyramid.append(temp_gau)
      return gaussian_pyramid
  def laplacian(self,gaussian_pyramid, up_times=5):
      laplacian_pyramid = [gaussian_pyramid[-1]]

      for i in range(up_times, 0, -1):
        # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
          temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
          temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
          laplacian_pyramid.append(temp_lap)
      return laplacian_pyramid

  def laplaceedge(self,img):
      g = numpy.array(((0, 1, 0), (1, -4, 1), (0, 1, 0)))
      # g = numpy.array(((1, 1, 1), (1, -8, 1), (1, 1, 1)))
      re = numpy.zeros_like(img)  # 生成与img相同shape的全0数组
      for i in range(1, img.shape[0] - 1):
          for j in range(1, img.shape[1] - 1):
              re[i, j] = (img[i - 1: i + 2, j - 1: j + 2] * g).sum()  # +img[i,j]
      # cv2.imwrite('./' + 'edge.jpg', re)
      return re
  def image_processing(self, x, sign, name):
      with tf.variable_scope(name):
          r, g, b = tf.split(x, num_or_size_splits=3, axis=-1)

          # normalize pixel with pre-calculated value
          rgb_mean = (0.4480, 0.4371, 0.4041)
          rgb_mean = tf.constant(rgb_mean, dtype=tf.float32)
          rgb = tf.concat([(r + sign * rgb_mean[0]),
                           (g + sign * [1]),
                           (b + sign * [2])], axis=-1)
          return rgb

  def channel_attention(self, x, f, reduction, name):
      """
      Channel Attention (CA) Layer
      :param x: input layer
      :param f: conv2d filter size
      :param reduction: conv2d filter reduction rate
      :param name: scope name
      :return: output layer
      """
      with tf.variable_scope("CA-%s" % name):
          skip_conn = tf.identity(x, name='identity')

          x = tfu.adaptive_global_average_pool_2d(x)

          x = tfu.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
          x = tf.nn.relu(x)

          x = tfu.conv2d(x, f=f, k=1, name="conv2d-2")

          x = tf.nn.sigmoid(x)
          return tf.multiply(skip_conn, x)
  def sa_attention(self, x, channels, name):
    with tf.variable_scope("SA-%s" % name):
      batch_size, height, width, num_channels = x.get_shape().as_list()
      f = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='f_conv')  # [bs, h, w, c']
      f = max_pooling(f)

      g = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='g_conv')  # [bs, h, w, c']

      h = conv(x, channels // 2, kernel=1, stride=1, sn=True, scope='h_conv')  # [bs, h, w, c]
      h = max_pooling(h)

      # N = h * w
      s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

      beta = tf.nn.softmax(s)  # attention map
      # beta = s  # attention map

      o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
      gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

      o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
      o = conv(o, channels, kernel=1, stride=1, sn=True, scope='attn_conv')
      x = gamma * o + x

    return x

  def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name):
      with tf.variable_scope("RCAB-%s" % name):
          skip_conn = tf.identity(x, name='identity')

          x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
          x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-1")(x) if use_bn else x
          x = tf.nn.relu(x)

          x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
          x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-2")(x) if use_bn else x
# #    #############
#           x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-3")
#           x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-3")(x) if use_bn else x
# #    #############
          x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)

          return 1 * x + skip_conn  # tf.math.add(self.res_scale * x, skip_conn)

  def residual_group(self, x, f, kernel_size, reduction, use_bn, name):
      with tf.variable_scope("RG-%s" % name):
          skip_conn = tf.identity(x, name='identity')

          for i in range(3):
              x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i))

          x = tfu.conv2d(x, f=f, k=kernel_size, name='rg-conv-1')
          # x = x + skip_conn

          return x + skip_conn # tf.math.add(x, skip_conn)

  def up_scaling(self, x, f, scale_factor, name):
      """
      :param x: image
      :param f: conv2d filter
      :param scale_factor: scale factor
      :param name: scope name
      :return:
      """
      with tf.variable_scope(name):
          if scale_factor == 3:
              x = tfu.conv2d(x, f * 9, k=1, name='conv2d-image_scaling-0')
              x = tfu.pixel_shuffle(x, 3)
          elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
              log_scale_factor = int(np.log2(scale_factor))
              for i in range(log_scale_factor):
                  x = tfu.conv2d(x, f * 4, k=1, name='conv2d-image_scaling-%d' % i)
                  x = tfu.pixel_shuffle(x, 2)
          else:
              raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
          return x

  def spatial_attention(self, input_feature, index):
      with tf.variable_scope("sp_attention_%s"%index):
          avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
          max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
          concat = tf.concat([avg_pool,max_pool], axis=3)
          spatial_layer = tf.layers.conv2d(inputs=concat, filters=1, kernel_size=(4,4),
                                           padding="same", activation=None)
          spatial_attention = tf.nn.sigmoid(spatial_layer)
          return input_feature*spatial_attention

  def upsample_and_concat(self, x1, x2, output_channels, in_channels):
      pool_size = 2
      deconv_filter = tf.Variable(
          tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
      deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

      deconv_output = tf.concat([deconv, x2], 3)
      deconv_output.set_shape([None, None, None, output_channels * 2])

      return deconv_output
  def dilatedConv2d(self,input,filter,stride,rate,padding,name):
      conv = tf.nn.dilation2d(input,filter,stride,rate,padding,name)
      return conv


  def model(self):

    with tf.variable_scope("main_branch") as scope3:
      input_raw = tf.concat(axis=3, values=[self.images])

      head = tfu.conv2d(input_raw, f=32, k=3, name="conv2d-head1")
      head = tfu.conv2d(head, f=64, k=3, name="conv2d-head2")

      x = head
      for i in range(3):
        x = self.residual_group(x, 64, 3, 16, False, name=str(i))

      body = tfu.conv2d(x, f=64, k=3, name="conv2d-body")

      body += x  # tf.math.add(body, head)

      body = tfu.conv2d(body, f=64, k=3, name="conv2d-tail")
      tail = tfu.conv2d(body, f=3, k=3, name="conv2d-tail2")

      return tail


  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
