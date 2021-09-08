import cv2
import numpy


from ops import *
import tensorflow.contrib.layers as tcl
import drcan_metric
import drcan_tfutil as tfu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as contrib_layers


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
                 test_image_name=None,
                 test_wb_name=None,
                 test_ce_name=None,
                 test_gc_name=None,
                 test_eh_name=None,
                 # test_sm_name=None,
                 id=None
                 ):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.dropout_keep_prob = 0.5
        self.test_image_name = test_image_name
        self.test_wb_name = test_wb_name
        self.test_ce_name = test_ce_name
        self.test_gc_name = test_gc_name
        self.test_eh_name = test_eh_name
        # self.test_sm_name = test_sm_name
        self.id = id
        self.c_dim = c_dim
        self.df_dim = 64
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                     name='images')
        self.pred_h = self.model()

        self.saver = tf.train.Saver()

    def train(self, config):

        # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
        image_test = get_image(self.test_image_name, is_grayscale=False)
        shape = image_test.shape
        expand_test = image_test[np.newaxis, :, :, :]
        expand_zero = np.zeros([self.batch_size - 1, shape[0], shape[1], shape[2]])
        batch_test_image = np.append(expand_test, expand_zero, axis=0)


        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        start_time = time.time()
        result_h = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image})
        all_time = time.time()
        final_time = all_time - start_time
        print(final_time)

        _, h, w, c = result_h.shape
        for id in range(0, 1):
            result_h0 = result_h[id, :, :, :].reshape(h, w, 3)
            result_h0 = result_h0.squeeze()
            image_path0 = os.path.join(os.getcwd(), config.sample_dir)
            prefix = self.test_image_name.split('.')[0]
            image_path = os.path.join(image_path0, prefix + '_out.png')
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

    def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
            x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-1")(x) if use_bn else x
            x = tf.nn.relu(x)

            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
            x = tf.layers.BatchNormalization(epsilon=1.1e-5, name="bn-2")(x) if use_bn else x

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
    ''''''

    def model(self):
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

            x = head
            for i in range(3):  
                x = self.residual_group(x, 64, 3, 16, False, name=str(i))

            x = self.up_scaling(x, 64, 2, name='up-scaling')
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
