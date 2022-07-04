from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D,Lambda, add
from keras.layers.core import Activation
from keras import backend as K
import tensorflow as tf

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat

def add(tensor_a, tensor_b):
    return tf.add(tensor_a, tensor_b)

def attention(tensor, att_tensor, n_filters=512, kernel_size=[1, 1]):
    xt = K.int_shape(att_tensor)
    g1 = Conv2D(n_filters, (1,1))(tensor)
    x1 = Conv2D(n_filters, (1,1))(att_tensor)
    net = add(g1, x1)
    net = LeakyReLU()(net)
    net  = Conv2D(n_filters, (1,1))(net)
    net  = Activation('sigmoid')(net)
    net = expend_as(net ,  3)
    #net = tf.concat([att_tensor, net], axis=-1)
    net = net * att_tensor
    return net