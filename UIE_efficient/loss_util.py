import tensorflow as tf


def l1_loss(gt, gen):
    """
     Absolute Difference loss between gt and gen.

    Args:
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: Weighted loss float Tensor, it is scalar.
    """
    return tf.losses.absolute_difference(labels=gt, predictions=gen, scope='l1_loss')

def ssim_loss(gt, gen):
    """
    Structural Similarity loss between gt and gen

    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :return: Loss
    """
    return 1 - tf.reduce_mean(
        tf.image.ssim(
            gen,
            gt,
            max_val=1))

def mse_loss(gt, gen):
    """
    l2 loss between gt and gen

    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :return: L2 loss
    """
    return tf.losses.mean_squared_error(
        labels=gt,
        predictions=gen)




def msssim_loss(gt, gen):
    """
    Computes the MS-SSIM loss between gt and gen

    Args:
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: Weighted loss float Tensor, it is scalar.
    """
    return 1 - tf.reduce_mean(
        tf.image.ssim_multiscale(
            img1=gen,
            img2=gt,
            max_val=1)
    )


def gdl_loss(gt, gen):
    """
    Compute the image gradient loss between gt and gen
    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :return: gdl_loss, it is scalar
    """
    dy_gt, dx_gt = tf.image.image_gradients(gt)
    dy_gen, dx_gen = tf.image.image_gradients(gen)
    grad_loss = tf.reduce_mean(tf.abs(dy_gen - dy_gt) + tf.abs(dx_gen - dx_gt))

    return grad_loss


def l2_l1_loss(gt, gen, alpha=0.8):
    """
    Loss function mix l1_loss and l2_loss

    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :param alpha: coefficient, default set as 0.8
    :return: Loss
    """
    l1 = l1_loss(gt, gen)
    l2 = mse_loss(gt, gen)

    return alpha * l2 + (1 - alpha) * l1


def ssim_l1_loss(gt, gen, alpha=0.8):
    """
    Loss function, calculating alpha * ssim_loss + (1-alpha) * l1_loss
    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :param alpha: coefficient, set to 0.8 according to paper
    :return: Loss
    """
    l1 = l1_loss(gt, gen)
    ssim_loss_ = ssim_loss(gt, gen)

    return alpha * ssim_loss_ + (1 - alpha) * l1


def msssim_l1_loss(gt, gen, alpha=0.8):
    """
    Loss function, calculating alpha * msssim_loss + (1-alpha) * l1_loss
    according to 'Underwater Color Restoration Using U-Net Denoising Autoencoder' [Yousif]
    :alpha: default value accoording to paper

    Args:
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :param alpha: coefficient, set to 0.8 according to paper
        :return: Loss
    """
    l1 = l1_loss(gt, gen)

    # ssim_multiscale already calculates the dyalidic pyramid (with as replacment avg.pooling)
    msssim_loss_ = msssim_loss(gt, gen)

    return alpha * msssim_loss_ + (1 - alpha) * l1


def gdl_l1_loss(gt, gen, alpha=0.8):
    """
    Loss function, calculating alpha * gdl_loss + (1-alpha) * l1_loss
    :param gt: The ground truth output tensor, same dimensions as 'gen'.
    :param gen: The predicted outputs.
    :param alpha: coefficient, set to 0.8 according to paper
    :return: Loss
    """
    l1 = l1_loss(gt, gen)
    gdl = gdl_loss(gt, gen)

    return alpha * gdl + (1 - alpha) * l1




