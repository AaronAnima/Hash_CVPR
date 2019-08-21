import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from config import flags
from utils import WeightNorm

def get_dwG(shape_z=(None, 100), shape_h=(0, 16)): # Dimension of gen filters in first conv layer. [64]
    s16 = flags.img_size_h // 16
    gf_dim = 64  # Dimension of gen filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    n_z = Input(shape_z)
    n_h = Input(shape_h)
    n = Concat(-1)([n_z, n_h])

    n = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init=w_init, act=tf.identity, b_init=None)(n)

    n = Reshape(shape=[-1, s16, s16, gf_dim * 8])(n)

    n = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)

    n = DeConv2d(gf_dim * 4, (5, 5), strides=(2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)

    n = DeConv2d(gf_dim * 2, (5, 5), strides=(2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)

    n = DeConv2d(gf_dim, (5, 5), strides=(2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=tf.nn.relu,  gamma_init=gamma_init)(n)

    n = DeConv2d(flags.c_dim, (5, 5), strides=(2, 2), act=tf.nn.tanh, W_init=w_init)(n)
    return tl.models.Model(inputs=[n_z, n_h], outputs=n, name='generator')


def get_dwD(shape): # Dimension of discrim filters in first conv layer. [64]

    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    ni = Input(shape)

    n = Conv2d(df_dim, (5, 5), (2, 2), act=lrelu, W_init=w_init)(ni)

    n = Conv2d(df_dim * 2, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)

    nf = Flatten(name='flatten')(n)

    n1 = Dense(n_units=1, act=tf.identity, W_init=w_init)(nf)

    n2 = Dense(n_units=flags.h_dim, act=tf.sigmoid, W_init=w_init, name='hash_layer')(nf)


    return tl.models.Model(inputs=ni, outputs=[nf, n1, n2], name='discriminator')
