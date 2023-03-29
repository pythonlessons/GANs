import tensorflow as tf
from keras.initializers import RandomNormal
from keras import layers

# Define the generator model
def build_generator(noise_dim, output_channels=3):
    inputs = layers.Input(shape=noise_dim, name="input")

    x = layers.Dense(4*4*512, use_bias=False)(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding="same", use_bias=False, 
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", use_bias=False, 
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False, 
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False, 
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(output_channels, (4, 4), strides=(2, 2), padding="same", use_bias=False, 
                               activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    assert x.shape == (None, 64, 64, output_channels)
    # x = layers.Activation('tanh')(x)

    # x = layers.Dense(7*7*256, use_bias=False)(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    # x = layers.Reshape((7, 7, 256))(x)
    # assert x.shape == (None, 7, 7, 256)  # Note: None is the batch size

    # x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    # assert x.shape == (None, 7, 7, 128)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    # x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    # assert x.shape == (None, 14, 14, 64)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)

    # x = layers.Conv2DTranspose(output_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    # # add last acvitaion layer of tanh
    # x = layers.Activation('tanh')(x)
    # assert x.shape == (None, 28, 28, output_channels)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model


# Define the discriminator model
def build_discriminator(img_shape, activation="linear"):
    inputs = layers.Input(shape=img_shape, name="input")

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(inputs)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)

    x = layers.Flatten()(x)
    # x = layers.Dropout(0.3)(x)

    # x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    # x = layers.LeakyReLU(0.2)(x)

    # x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    # x = layers.LeakyReLU(0.2)(x)

    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation=activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model