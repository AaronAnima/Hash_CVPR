import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags


def get_dataset_train():
    if flags.dataset == 'MNIST':
        X_train, _, _, _, X_test, _ = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        X_train = X_train * 2 - 1
    if flags.dataset == 'CIFAR_10':
        X_train, _, X_test, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
        X_train = X_train / 127.5 - 1

    def generator_train():
        for image in X_train:
            yield image

    def _map_fn(image):
        image = tf.image.random_flip_left_right(image)
        return image

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.float32)
    ds = train_ds.shuffle(buffer_size=4096)
    ds = ds.repeat(flags.n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=4) # For concurrency
    return ds, X_train.shape[0]


def get_dataset_eval():
    if flags.dataset == 'MNIST':
        X_train, Y_train, _, _, X_test, _ = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        X_train = X_train * 2 - 1
    if flags.dataset == 'CIFAR_10':
        X_train, Y_train, X_test, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
        X_train = X_train / 127.5 - 1

    def generator():
        for image, label in zip(X_train, Y_train):
            yield image, label

    def _map_fn(image, label):
        return image, label

    ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))

    ds = ds.map(_map_fn, num_parallel_calls=4)

    ds = ds.batch(flags.batch_size_eval)

    ds = ds.prefetch(buffer_size=4) # For concurrency

    return ds


