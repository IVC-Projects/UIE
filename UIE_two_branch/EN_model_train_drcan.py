from tensorflow import tanh

from loss_util import gdl_loss, mse_loss, ssim_loss
# from loss_util import ssim_loss
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
        self.dropout_keep_prob = 0.5

        self.c_dim = c_dim
        self.df_dim = 64
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.vgg_dir = './vgg_pretrained/imagenet-vgg-verydeep-19.mat'
        self.CONTENT_LAYER = 'relu5_4'
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                     name='images')

        self.labels_image = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                           name='labels_image')

        self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],
                                          name='images_test')

        self.labels_test = tf.placeholder(tf.float32, [1, self.label_height, self.label_width, self.c_dim],
                                          name='labels_test')

        self.pred_h1 = self.model()

        self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
        self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image * 255))
        self.loss_texture1 = tf.reduce_mean(
            tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER] - self.labels_texture_vgg[self.CONTENT_LAYER]))

        self.ssim_loss = ssim_loss(self.labels_image, self.pred_h1)
        self.gdl_loss = gdl_loss(self.labels_image, self.pred_h1)
        self.mse_loss = mse_loss(self.labels_image, self.pred_h1)
        self.loss = 0.05 * self.loss_texture1 + self.mse_loss + self.ssim_loss + 0.01 * self.gdl_loss  ### final


        t_vars = tf.trainable_variables()

        self.saver = tf.train.Saver(max_to_keep=50)

    def train(self, config):
        if config.is_train:
            data_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/input_train")
            image_train_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/gt_train")
            data_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/input_test")
            image_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/gt_test")

            seed = 568
            np.random.seed(seed)
            np.random.shuffle(data_train_list)
            np.random.seed(seed)
            np.random.shuffle(image_train_list)

        else:
            data_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/input_test")
            image_test_list = prepare_data(self.sess, dataset="./train_and_test_dataset_euvp_uwscences/gt_test")

        sample_data_files = data_test_list[16:20]
        sample_image_files = image_test_list[16:20]

        sample_data = [
            get_image(sample_data_file,
                      is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
        sample_lable_image = [
            get_image(sample_image_file,
                      is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

        sample_inputs_data = np.array(sample_data).astype(np.float32)
        sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

        self.train_op = tf.train.AdamOptimizer(config.learning_rate, 0.9).minimize(self.loss)
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
                lr=0.001
                if ep<200:
                    config.learning_rate=lr
                elif ep>=200 and ep<400:
                    config.learning_rate=lr/2
                elif ep>=400 and ep<600:
                    config.learning_rate=lr/5
                elif ep>=600 and ep<800:
                    config.learning_rate=lr/10
                else:
                    config.learning_rate=lr/20
                print("learning_rate",config.learning_rate)

                batch_idxs = len(data_train_list) // config.batch_size
                for idx in range(0, batch_idxs):

                    batch_files = data_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_image_files = image_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                    batch_ = [
                        get_image(batch_file,
                                  is_grayscale=self.is_grayscale) for batch_file in batch_files]
                    batch_labels_image = [
                        get_image(batch_image_file,
                                  is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]

                    batch_input = np.array(batch_).astype(np.float32)
                    batch_image_input = np.array(batch_labels_image).astype(np.float32)

                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss],
                                           feed_dict={self.images: batch_input,
                                                      self.labels_image: batch_image_input})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, err))

                    if idx == batch_idxs - 1:
                        batch_test_idxs = len(data_test_list) // config.batch_size
                        err_test = np.ones(batch_test_idxs)
                        for idx_test in range(0, batch_test_idxs):
                            sample_data_files = data_train_list[
                                                idx_test * config.batch_size:(idx_test + 1) * config.batch_size]
                            sample_image_files = image_train_list[
                                                 idx_test * config.batch_size: (idx_test + 1) * config.batch_size]

                            sample_data = [get_image(sample_data_file,
                                                     is_grayscale=self.is_grayscale) for sample_data_file in
                                           sample_data_files]

                            sample_lable_image = [get_image(sample_image_file,
                                                            is_grayscale=self.is_grayscale) for sample_image_file in
                                                  sample_image_files]

                            sample_inputs_data = np.array(sample_data).astype(np.float32)
                            sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

                            err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data,

                                                                                     self.labels_image: sample_inputs_lable_image})

                        loss[ep] = np.mean(err_test)
                        print(loss)
                        self.save(config.checkpoint_dir, counter)

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
            print("ca_shape_1",x)
            x = tf.nn.relu(x)

            x = tfu.conv2d(x, f=f, k=1, name="conv2d-2")
            print("ca_shape_2", x)

            x = tf.nn.sigmoid(x)
            x = tf.multiply(skip_conn, x)

            return x

    def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
            print("rcab_shape_1:",x)
            x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-1")(x) if use_bn else x
            x = tf.nn.relu(x)

            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
            print("rcab_shape_2:", x)
            x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-2")(x) if use_bn else x

            x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)
            rcab = 1 * x + skip_conn
            print("rcab_shape:",rcab)

            return rcab  # tf.math.add(self.res_scale * x, skip_conn)

    def residual_group(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RG-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            for i in range(3):
                x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i))
                # print("x_shape_",x)

            x = tfu.conv2d(x, f=f, k=kernel_size, name='rg-conv-1')
            print("rg_x_shape",x)
            # x = x + skip_conn
            rg = x + skip_conn
            # print("rg_shape",rg)
            return rg  # tf.math.add(x, skip_conn)

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
                print("up_shape:",x)
                x = tfu.pixel_shuffle(x, 3)
            elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
                log_scale_factor = int(np.log2(scale_factor))
                for i in range(log_scale_factor):
                    x = tfu.conv2d(x, f * 4, k=1, name='conv2d-image_scaling-%d' % i)
                    print("up_shape:", x)
                    x = tfu.pixel_shuffle(x, 2)
            else:
                raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
            return x

    def model(self):  # best
        with tf.variable_scope("main_branch") as scope3:
            input_raw = tf.concat(axis=3, values=[self.images])
            #    branch 2
            input = tf.concat(axis=3, values=[self.images])
            out_conv1 = tf.nn.relu(conv2d(input, 3, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out1"))
            out_conv2 = tf.nn.relu(conv2d(out_conv1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out2"))
            out_conv3 = tf.nn.relu(conv2d(out_conv2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out3"))
            out_concat1 = tf.concat(axis=3, values=[out_conv1, out_conv2, out_conv3, input])
            out_conv4 = tf.nn.relu(conv2d(out_concat1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out4"))
            out_conv5 = tf.nn.relu(conv2d(out_conv4, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out5"))
            out_conv6 = tf.nn.relu(conv2d(out_conv5, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out6"))
            out_concat2 = tf.concat(axis=3, values=[out_concat1, out_conv4, out_conv5, out_conv6])
            out_conv7 = tf.nn.relu(conv2d(out_concat2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out7"))
            out_conv8 = tf.nn.relu(conv2d(out_conv7, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out8"))
            out_conv9 = tf.nn.relu(conv2d(out_conv8, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out9"))
            out_concat3 = tf.concat(axis=3, values=[out_concat2, out_conv7, out_conv8, out_conv9])
            out_conv10 = conv2d(out_concat3, 16, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d_out10")
            out1 = tf.nn.sigmoid(out_conv10)

            #   branch 1

            head = tfu.conv2d(input_raw, f=64, k=3, name="conv2d-head1")
            head = tfu.conv2d(head, f=64, k=3, s=2, name="conv2d-head2")
            print('head:', head)  # 128x128

            x = head
            for i in range(3):  
                x = self.residual_group(x, 64, 3, 16, False, name=str(i))


            tail = tfu.conv2d(x, f=3, k=3, name="conv2d-tail")


            x = tf.nn.sigmoid(tail)
            out2 = x
            input_concat = tf.concat(axis=3, values=[out1, out2])  # best
            conv1_concat = conv2d(input_concat, 6, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv1_concat")
            conv2_concat = conv2d(conv1_concat, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_concat")
            conv3_concat = conv2d(conv2_concat, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_concat")
            conv4_concat = conv2d(conv3_concat, 16, 6, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_s_concat")

            conv_s_concat = tf.nn.sigmoid(conv4_concat)


            final_out = tf.multiply(0.5, tf.add(tf.multiply(out1, conv_s_concat[:, :, :, 0:3]),
                                                tf.multiply(out2, conv_s_concat[:, :, :, 3:6])))
            return final_out
    
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
