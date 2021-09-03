import numpy
from tensorflow import tanh

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
import tensorflow.contrib.layers as tcl
import drcan_metric
import drcan_tfutil as tfu
import tensorflow.contrib.slim as slim


class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=460,
               image_width=620,
               label_height=460, 
               label_width=620,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None, 
               sample_dir=None,
               test_image_name = None,
               test_wb_name = None,
               test_ce_name = None,
               test_gc_name = None,
               test_eh_name = None,
               test_sm_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    self.test_wb_name = test_wb_name
    self.test_ce_name = test_ce_name
    self.test_gc_name = test_gc_name
    self.test_eh_name = test_eh_name
    self.test_sm_name = test_sm_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.images_eh = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],name='images_eh')
    self.images_sm = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],name='images_sm')
    self.pred_h = self.model()


    self.saver = tf.train.Saver()
     
  def train(self, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)

    # wb_test =  get_image(self.test_wb_name,is_grayscale=False)
    # shape = wb_test.shape
    # expand_test = wb_test[np.newaxis,:,:,:]
    # expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    # batch_test_wb = np.append(expand_test,expand_zero,axis = 0)
    #
    # ce_test =  get_image(self.test_wb_name,is_grayscale=False)
    # shape = ce_test.shape
    # expand_test = ce_test[np.newaxis,:,:,:]
    # expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    # batch_test_ce = np.append(expand_test,expand_zero,axis = 0)
    #
    # gc_test =  get_image(self.test_wb_name,is_grayscale=False)
    # shape = gc_test.shape
    # expand_test = gc_test[np.newaxis,:,:,:]
    # expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    # batch_test_gc = np.append(expand_test,expand_zero,axis = 0)
    #
    # eh_test = get_image(self.test_wb_name, is_grayscale=False)
    # shape = eh_test.shape
    # expand_test = eh_test[np.newaxis, :, :, :]
    # expand_zero = np.zeros([self.batch_size - 1, shape[0], shape[1], shape[2]])
    # batch_test_eh = np.append(expand_test, expand_zero, axis=0)

    #sm_test = get_image(self.test_sm_name, is_grayscale=False)
    #shape = sm_test.shape
    #expand_test = sm_test[np.newaxis, :, :, :]
    expand_zero = np.zeros([self.batch_size - 1, shape[0], shape[1], shape[2]])
    batch_test_sm = np.append(expand_test, expand_zero, axis=0)


    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()   
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image,self.images_sm: batch_test_sm})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)


    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h0 = result_h[id,:,:,:].reshape(h , w , 3)
        result_h0 = result_h0.squeeze()
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        prefix = self.test_image_name.split('.')[0]
        image_path = os.path.join(image_path0, prefix+'_out.png')
        imsave_lable(result_h0, image_path)

  def image_processing(self, x, sign, name):
    with tf.variable_scope(name):
      r, g, b = tf.split(x, num_or_size_splits=3, axis=-1)

      # normalize pixel with pre-calculated value based on DIV2K DataSet
      rgb_mean = (0.4480, 0.4371, 0.4041)
      rgb_mean = tf.constant(rgb_mean, dtype=tf.float32)
      rgb = tf.concat([(r + sign * rgb_mean[0]),
                       (g + sign * [1]),
                       (b + sign * [2])], axis=-1)
      return rgb
  def laplaceedge(self,img):
      g = numpy.array(((0, 1, 0), (1, -4, 1), (0, 1, 0)))
      # g = numpy.array(((1, 1, 1), (1, -8, 1), (1, 1, 1)))
      re = numpy.zeros_like(img)  # 生成与img相同shape的全0数组
      for i in range(1, img.shape[0] - 1):
          for j in range(1, img.shape[1] - 1):
              re[i, j] = (img[i - 1: i + 2, j - 1: j + 2] * g).sum()  # +img[i,j]
      # cv2.imwrite('./' + 'edge.jpg', re)
      return re

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

      # ###########
      # x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-3")
      # x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-3")(x) if use_bn else x
      # ##########

      x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)
      return 1 * x + skip_conn  # tf.math.add(self.res_scale * x, skip_conn)

  def residual_group(self, x, f, kernel_size, reduction, use_bn, name):
    with tf.variable_scope("RG-%s" % name):
      skip_conn = tf.identity(x, name='identity')

      for i in range(3):
        x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i))

      x = tfu.conv2d(x, f=f, k=kernel_size, name='rg-conv-1')
      return x + skip_conn  # tf.math.add(x, skip_conn)
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
  def upsample_and_concat(self, x1, x2, output_channels, in_channels):
      pool_size = 2
      deconv_filter = tf.Variable(
          tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
      deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

      deconv_output = tf.concat([deconv, x2], 3)
      deconv_output.set_shape([None, None, None, output_channels * 2])

      return deconv_output

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
