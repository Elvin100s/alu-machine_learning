#!/usr/bin/env python3
"""Module that defines the NST class, which performs neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer

    Public class attributes:
        style_layers: the VGG19 layers used for style extraction
        content_layer: the VGG19 layer used for content extraction
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the instance
        Args:
            style_image: np.ndarray of shape (h, w, 3) with the style image
            content_image: np.ndarray of shape (h, w, 3) with the content image
            alpha: the weight for content cost
            beta: the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or \
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        if not tf.executing_eagerly():
            enable = getattr(tf, 'enable_eager_execution', None)
            if enable is None:
                enable = tf.compat.v1.enable_eager_execution
            enable()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image so its pixels are in [0, 1] and its largest
        side is 512 pixels

        Args:
            image: np.ndarray of shape (h, w, 3) containing the image

        Returns:
            the scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or \
                image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w = image.shape[0], image.shape[1]
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = np.expand_dims(image, axis=0)
        resize = getattr(tf.image, 'resize_bicubic', None)
        if resize is None:
            resize = getattr(tf.compat.v1.image, 'resize_bicubic', None)
        if resize is None:
            resized = tf.image.resize(image, (h_new, w_new),
                                      method='bicubic')
        else:
            resized = resize(image, (h_new, w_new))
        resized = resized / 255
        return tf.clip_by_value(resized, 0, 1)

    def load_model(self):
        """Creates the model used to calculate cost and saves it in the
        instance attribute model
        """
        base = tf.keras.applications.VGG19(include_top=False,
                                           weights='imagenet')
        config = base.get_config()
        for layer in config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
        vgg = tf.keras.models.Model.from_config(config)
        vgg.set_weights(base.get_weights())
        for layer in vgg.layers:
            layer.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(vgg.input, outputs)
