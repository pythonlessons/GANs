import os
import cv2
import typing
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from keras.callbacks import TensorBoard

from model import build_generator, build_discriminator


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

class GAN(tf.keras.models.Model):
    """A Generative Adversarial Network (GAN) implementation.

    This class inherits from `tf.keras.models.Model` and provides a
    straightforward implementation of the GAN algorithm.
    """
    def __init__(
            self, 
            discriminator: tf.keras.models.Model, 
            generator: tf.keras.models.Model, 
            noise_dim: int
        ) -> None:
        """Initializes the GAN class.

        Args:
            discriminator (tf.keras.models.Model): A `tf.keras.model.Model` instance that acts
                as the discriminator in the GAN algorithm.
            generator (tf.keras.models.Model): A `tf.keras.model.Model` instance that acts as
                the generator in the GAN algorithm.
            noise_dim (int): The dimensionality of the noise vector that is
                inputted to the generator.
        """
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim

    def compile(
            self, 
            discriminator_opt: tf.keras.optimizers.Optimizer,
            generator_opt: tf.keras.optimizers.Optimizer, 
            discriminator_loss: typing.Callable,
            generator_loss: typing.Callable, 
            **kwargs
        ) -> None:
        """Configures the model for training.

        Args:
            discriminator_opt (tf.keras.optimizers.Optimizer): The optimizer for the discriminator.
            generator_opt (tf.keras.optimizers.Optimizer): The optimizer for the generator.
            discriminator_loss (typing.Callable): The loss function for the discriminator.
            generator_loss (typing.Callable): The loss function for the generator.
        """
        super(GAN, self).compile(**kwargs)
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

    def train_step(self, real_images: tf.Tensor) -> typing.Dict[str, tf.Tensor]:
        """Executes one training step and returns the loss.
        
        Args:
            real_images (tf.Tensor): A batch of real images from the dataset.
            
        Returns:
            typing.Dict[str, tf.Tensor]: A dictionary of metric values and losses.
        """
        batch_size = tf.shape(real_images)[0]
        # Generate random noise for the generator
        noise = tf.random.normal([batch_size, self.noise_dim])

        # Train the discriminator with both real images (label as 1) and fake images (label as 0) 
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images using the generator
            generated_images = self.generator(noise, training=True)

            # Get the discriminator's predictions for real and fake images
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate generator and discriminator losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Calculate gradients of generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients to generator and discriminator optimizer
        self.generator_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Update the metrics.
        self.compiled_metrics.update_state(real_output, fake_output)

        # Construct a dictionary of metric results and losses
        results = {m.name: m.result() for m in self.metrics}
        results.update({"d_loss": disc_loss, "g_loss": gen_loss})

        return results


class ResultsCallback(tf.keras.callbacks.Callback):
    """A callback that saves generated images after each epoch."""
    def __init__(
            self, 
            noise_dim: int, 
            results_path: str, 
            examples_to_generate: int=16, 
            grid_size: tuple=(4, 4), 
            spacing: int=5, 
            gif_size: tuple=(416, 416), 
            duration: float=0.1
        ):
        """ Initializes the ResultsCallback class.

        Args:
            noise_dim (int): The dimensionality of the noise vector that is inputted to the generator.
            results_path (str): The path to the directory where the results will be saved.
            examples_to_generate (int, optional): The number of images to generate and save. Defaults to 16.
            grid_size (tuple, optional): The size of the grid to arrange the generated images. Defaults to (4, 4).
            spacing (int, optional): The spacing between the generated images. Defaults to 5.
            gif_size (tuple, optional): The size of the gif to be generated. Defaults to (416, 416).
            duration (float, optional): The duration of each frame in the gif. Defaults to 0.1.
        """
        super(ResultsCallback, self).__init__()
        self.seed = tf.random.normal([examples_to_generate, noise_dim])
        self.results = []
        self.results_path = results_path + '/results'
        self.grid_size = grid_size
        self.spacing = spacing
        self.gif_size = gif_size
        self.duration = duration

        # create the results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)

    def save_pred(self, epoch: int, results: list) -> None:
        """ Saves the generated images as a grid and as a gif.

        Args:
            epoch (int): The current epoch.
            results (list): A list of generated images.
        """
        # construct an image from generated images with spacing between them using numpy
        w, h , c = results[0].shape
        # construct grid with self.grid_size
        grid = np.zeros((self.grid_size[0] * w + (self.grid_size[0] - 1) * self.spacing, self.grid_size[1] * h + (self.grid_size[1] - 1) * self.spacing, c), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid[i * (w + self.spacing):i * (w + self.spacing) + w, j * (h + self.spacing):j * (h + self.spacing) + h] = results[i * self.grid_size[1] + j]

        # save the image
        cv2.imwrite(f'{self.results_path}/img_{epoch}.png', grid)

        # save image to memory resized to gif size
        self.results.append(cv2.resize(grid, self.gif_size, interpolation=cv2.INTER_AREA))

    def on_epoch_end(self, epoch: int, logs: dict=None) -> None:
        """Executes at the end of each epoch."""
        predictions = self.model.generator(self.seed, training=False)
        predictions_uint8 = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
        self.save_pred(epoch, predictions_uint8)

    def on_train_end(self, logs=None) -> None:
        """Executes at the end of training."""
        # save the results as a gif with imageio

        # Create a list of imageio image objects from the OpenCV images
        imageio_images = [imageio.core.util.Image(image) for image in self.results]

        # Write the imageio images to a GIF file
        imageio.mimsave(self.results_path + "/output.gif", imageio_images, duration=self.duration)


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [-1, 1]
x_train = (x_train.astype('float32') - 127.5) / 127.5 

# Set the input shape and size for the generator and discriminator
img_shape = (28, 28, 1) # The shape of the input image, input to the discriminator
noise_dim = 100 # The dimension of the noise vector, input to the generator
model_path = 'Models/01_GANs_introduction'
os.makedirs(model_path, exist_ok=True)

generator = build_generator(noise_dim)
discriminator = build_discriminator(img_shape)

generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)

callback = ResultsCallback(noise_dim=noise_dim, results_path=model_path)
tb_callback = TensorBoard(model_path + '/logs', update_freq=1)

gan = GAN(discriminator, generator, noise_dim)
gan.compile(discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss, run_eagerly=False)

gan.fit(x_train, epochs=100, batch_size=128, callbacks=[callback, tb_callback])