import tensorflow as tf
from keras import layers

# Define the generator model
def build_generator(noise_dim, output_channels=3, activation="tanh", alpha=0.2):
    inputs = layers.Input(shape=noise_dim, name="input")

    x = layers.Dense(4*4*512, use_bias=False)(inputs)

    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(output_channels, (5, 5), strides=(1, 1), padding="same", activation=activation, use_bias=False, dtype='float32')(x)
    assert x.shape == (None, 64, 64, output_channels)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

    
# Define the discriminator model
def build_discriminator(img_shape, activation='linear', alpha=0.2):
    inputs = layers.Input(shape=img_shape, name="input")

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1, activation=activation, dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model