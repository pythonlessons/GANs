import tensorflow as tf
from keras import layers

# Define the generator model
def build_generator(noise_dim, output_channels=3, alpha=0.2):
    inputs = layers.Input(shape=noise_dim, name="input")

    x = layers.Dense(4*4*512)(inputs)

    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(output_channels, (5, 5), strides=(1, 1), padding="same")(x)
    x = layers.Activation("tanh")(x)
    assert x.shape == (None, 64, 64, output_channels)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

    
# Define the discriminator model
def build_discriminator(img_shape, activation=None, alpha=0.2):
    inputs = layers.Input(shape=img_shape, name="input")

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1, activation=activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model