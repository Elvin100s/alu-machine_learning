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

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initializes the instance
        Args:
            style_image: np.ndarray of shape (h, w, 3) with the style image
            content_image: np.ndarray of shape (h, w, 3) with the content image
            alpha: the weight for content cost
            beta: the weight for style cost
            var: the weight for the variational cost
        """
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        style_h, style_w, style_c = style_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        content_h, content_w, content_c = content_image.shape
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        if (type(var) is not float and type(var) is not int) \
                or var < 0:
            raise TypeError('var must be a non-negative number')

        if not tf.executing_eagerly():
            enable = getattr(tf, 'enable_eager_execution', None)
            if enable is None:
                enable = tf.compat.v1.enable_eager_execution
            enable()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image so its pixels are in [0, 1] and its largest
        side is 512 pixels

        Args:
            image: np.ndarray of shape (h, w, 3) containing the image

        Returns:
            the scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

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
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        try:
            base.save('vgg_base_model')
            vgg = tf.keras.models.load_model('vgg_base_model',
                                             custom_objects=custom_objects)
        except Exception:
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

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            a tf.Tensor of shape (1, c, c) containing the gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        product = tf.einsum('bhwi,bhwj->bij', input_layer, input_layer)
        shape = tf.shape(input_layer)
        nb_locations = tf.cast(shape[1] * shape[2], tf.float32)
        return product / nb_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost and
        saves them in gram_style_features and content_feature
        """
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255)
        content_input = vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_input)[:-1]
        content_output = self.model(content_input)[-1]

        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer

        Args:
            style_output: tf.Tensor of shape (1, h, w, c) with the layer
                style output of the generated image
            gram_target: tf.Tensor of shape (1, c, c) with the gram matrix
                of the target style output for that layer

        Returns:
            the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        c = int(style_output.shape[-1])
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'.format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for the generated image

        Args:
            style_outputs: a list of tf.Tensor style outputs for the
                generated image

        Returns:
            the style cost
        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != length:
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    length))

        weight = 1 / length
        cost = 0
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            cost += weight * self.layer_style_cost(style_output, gram_target)
        return cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image

        Args:
            content_output: a tf.Tensor with the content output for the
                generated image

        Returns:
            the content cost
        """
        shape = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != shape:
            raise TypeError(
                'content_output must be a tensor of shape {}'.format(shape))

        return tf.reduce_mean(tf.square(content_output -
                                        self.content_feature))

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image

        Returns:
            the variational cost
        """
        vertical = tf.abs(generated_image[:, 1:, :, :] -
                          generated_image[:, :-1, :, :])
        horizontal = tf.abs(generated_image[:, :, 1:, :] -
                            generated_image[:, :, :-1, :])
        return tf.reduce_sum(vertical) + tf.reduce_sum(horizontal)

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image

        Returns:
            (J, J_content, J_style, J_var)
        """
        shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != shape:
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    shape))

        vgg19 = tf.keras.applications.vgg19
        generated_input = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(generated_input)

        J_style = self.style_cost(outputs[:-1])
        J_content = self.content_cost(outputs[-1])
        J_var = self.variational_cost(generated_image)
        J = self.alpha * J_content + self.beta * J_style + self.var * J_var
        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image

        Returns:
            gradients, J_total, J_content, J_style, J_var
        """
        shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != shape:
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image)

        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transfered image

        Args:
            iterations: the number of iterations of gradient descent
            step: if not None, the step at which to print information
            lr: the learning rate for gradient descent
            beta1: the beta1 parameter for gradient descent
            beta2: the beta2 parameter for gradient descent

        Returns:
            generated_image, cost
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be positive')
        if step is not None:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step >= iterations:
                raise ValueError(
                    'step must be positive and less than iterations')
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError('lr must be a number')
        if lr <= 0:
            raise ValueError('lr must be positive')
        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')
        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 must be in the range [0, 1]')
        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')
        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        if hasattr(tf, 'contrib'):
            generated_image = tf.contrib.eager.Variable(self.content_image)
            optimizer = tf.train.AdamOptimizer(lr, beta1, beta2)
        else:
            generated_image = tf.Variable(self.content_image)
            optimizer = tf.optimizers.Adam(lr, beta1, beta2)

        best_cost = float('inf')
        best_image = generated_image.numpy()

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)

            if float(J_total) < best_cost:
                best_cost = float(J_total)
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}, var {}"
                      .format(i, float(J_total), float(J_content),
                              float(J_style), float(J_var)))

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                clipped = tf.clip_by_value(generated_image, 0, 1)
                generated_image.assign(clipped)

        return best_image[0], best_cost
