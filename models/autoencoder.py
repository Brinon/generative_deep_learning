import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Flatten,
    Dense,
    Conv2DTranspose,
    Reshape,
    Lambda,
    Activation,
    BatchNormalization,
    LeakyReLU,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import plot_model

from utils.callbacks import step_decay_schedule


import numpy as np
import json
import os
import pickle


class Autoencoder:
    def __init__(
        self,
        input_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        z_dim,
        use_batch_norm=False,
        use_dropout=False,
    ):

        self.name = "autoencoder"

        self.input_dim = input_dim
        # number of conv filters of the encoder
        self.encoder_conv_filters = encoder_conv_filters
        # size of the conv filters of the encoder
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        # stride of the convs of the encode
        self.encoder_conv_strides = encoder_conv_strides
        # number of conv filters in the transpose convolutions of the decoder
        # last conv filter should match the last dimension of the input
        self.decoder_conv_t_filters = decoder_conv_t_filters
        # size of the transpose conv filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        # strides of the transpose conv filters
        self.decoder_conv_t_strides = decoder_conv_t_strides
        # sizse of latent representation
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()

    @classmethod
    def build_default(cls):
        """build the model with default params"""
        return cls(
            input_dim=(28, 28, 1),
            encoder_conv_filters=[32, 64, 64, 64],
            encoder_conv_kernel_size=[3, 3, 3, 3],
            encoder_conv_strides=[1, 2, 2, 1],
            decoder_conv_t_filters=[64, 64, 32, 1],
            decoder_conv_t_kernel_size=[3, 3, 3, 3],
            decoder_conv_t_strides=[1, 2, 2, 1],
            z_dim=2,
        )

    def _build(self):

        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name="encoder_input")

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding="same",
                name="encoder_conv_" + str(i),
            )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = tf.int_shape(x)[1:]

        x = Flatten()(x)
        encoder_output = Dense(self.z_dim, name="encoder_output")(x)

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER
        decoder_input = Input(shape=(self.z_dim,), name="decoder_input")

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding="same",
                name="decoder_conv_t_" + str(i),
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation("sigmoid")(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate=learning_rate)

        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, "viz"))
            os.makedirs(os.path.join(folder, "weights"))
            os.makedirs(os.path.join(folder, "images"))

        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump(
                [
                    self.input_dim,
                    self.encoder_conv_filters,
                    self.encoder_conv_kernel_size,
                    self.encoder_conv_strides,
                    self.decoder_conv_t_filters,
                    self.decoder_conv_t_kernel_size,
                    self.decoder_conv_t_strides,
                    self.z_dim,
                    self.use_batch_norm,
                    self.use_dropout,
                ],
                f,
            )

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(
        self,
        x_train,
        batch_size,
        epochs,
        run_folder,
        print_every_n_batches=100,
        initial_epoch=0,
        lr_decay=1,
    ):

        custom_callback = CustomCallback(
            run_folder, print_every_n_batches, initial_epoch, self
        )
        lr_sched = step_decay_schedule(
            initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1
        )

        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, "weights/weights.h5"),
            save_weights_only=True,
            verbose=1,
        )

        callbacks_list = [checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list,
        )

    def plot_model(self, run_folder):
        plot_model(
            self.model,
            to_file=os.path.join(run_folder, "viz/model.png"),
            show_shapes=True,
            show_layer_names=True,
        )
        plot_model(
            self.encoder,
            to_file=os.path.join(run_folder, "viz/encoder.png"),
            show_shapes=True,
            show_layer_names=True,
        )
        plot_model(
            self.decoder,
            to_file=os.path.join(run_folder, "viz/decoder.png"),
            show_shapes=True,
            show_layer_names=True,
        )


class VariationalAutoencoder(Autoencoder):
    # Instead of representing the latent space as 2 values, represents it as
    # a probability distribution from where we sample to decode

    def _build(self):

        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name="encoder_input")

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding="same",
                name="encoder_conv_" + str(i),
            )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = x.get_shape().as_list()[1:]
        x = Flatten()(x)

        # 2 heads, mu (center of the distribution) and logarithmic variance
        self.mu = Dense(self.z_dim, name="mu")(x)
        self.log_var = Dense(self.z_dim, name="log_var")(x)

        # encodes an input image into a probability distribution
        encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        # output for mu and log_var
        mu_log_var_concat = tf.keras.layers.Concatenate(
            axis=-1, name="encoder_output_concat"
        )([self.mu, self.log_var])

        @tf.function
        def sample_from_distribution(args):
            mu, log_var = args
            epsilon = tf.random.normal(shape=tf.shape(mu), mean=0, stddev=1.0)
            return mu + tf.exp(log_var / 2) * epsilon

        # the output of the decoder consist on sampling from the distribution
        encoder_output = Lambda(
            sample_from_distribution,
            name="encoder_output",
        )([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        # Decoder
        decoder_input = Input(shape=(self.z_dim,), name="decoder_input")

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding="same",
                name="decoder_conv_t_" + str(i),
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation("sigmoid")(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output, name="decoder")

        ### THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        # model with 3 outputs, the reconstructed output and the intermediate
        # layers tha represent the latent space
        self.model = Model(model_input, [model_output, mu_log_var_concat])

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate=learning_rate)

        def reconstruction_loss(y_true, y_pred):
            r_loss = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss * r_loss_factor

        def divergence_loss(y_true, y_pred):
            """measuers KL divergence between the predicted distribution"""
            # first half of y_true is mu, the other half is log_var
            print(f'ypred: {y_pred}')
            shape = y_pred.get_shape().as_list()[0]
            print(f'prediction shape: {shape}')
            idx = shape // 2

            mu = y_pred[:idx]
            log_var = y_pred[idx:]

            kl_loss = -0.5 * tf.math.reduce_sum(
                1 + log_var - tf.math.square(mu) - tf.math.exp(log_var),
                axis=1,
            )
            return kl_loss

        def vae_loss(y_true, y_pred):
            return reconstruction_loss(y_true, y_pred) + divergence_loss(y_true, y_pred)
            # return reconstruction_loss(y_true, y_pred)

        self.model.compile(
            optimizer=optimizer,
            loss={
                "decoder": reconstruction_loss,
                "encoder_output_concat": divergence_loss,
            },
        )
