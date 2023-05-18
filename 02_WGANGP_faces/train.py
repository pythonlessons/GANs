import os
import cv2
import typing
import imageio
import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from model import build_generator, build_discriminator

from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class WGAN_GP(tf.keras.models.Model):
    def __init__(
            self, 
            discriminator: tf.keras.models.Model, 
            generator: tf.keras.models.Model, 
            noise_dim: int, 
            discriminator_extra_steps: int=5, 
            gp_weight: typing.Union[float, int]=10.0
        ) -> None:
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(
            self, 
            discriminator_opt: tf.keras.optimizers.Optimizer, 
            generator_opt: tf.keras.optimizers.Optimizer, 
            discriminator_loss: typing.Callable, 
            generator_loss: typing.Callable, 
            **kwargs
        ) -> None:
        super(WGAN_GP, self).compile(**kwargs)
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

    def add_instance_noise(self, x: tf.Tensor, stddev: float=0.1) -> tf.Tensor:
        """ Adds instance noise to the input tensor."""
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=stddev, dtype=x.dtype)
        return x + noise

    def gradient_penalty(
            self, 
            real_samples: tf.Tensor, 
            fake_samples: tf.Tensor, 
            discriminator: tf.keras.models.Model
        ) -> tf.Tensor:
        """ Calculates the gradient penalty.

        Gradient penalty is calculated on an interpolated data
        and added to the discriminator loss.
        """
        batch_size = tf.shape(real_samples)[0]

        # Generate random values for epsilon
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

        # 1. Interpolate between real and fake samples
        interpolated_samples = epsilon * real_samples + ((1 - epsilon) * fake_samples)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            # 2. Get the Critic's output for the interpolated image
            logits = discriminator(interpolated_samples, training=True)

        # 3. Calculate the gradients w.r.t to the interpolated image
        gradients = tape.gradient(logits, interpolated_samples)

        # 4. Calculate the L2 norm of the gradients.
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))

        # 5. Calculate gradient penalty
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)

        return gradient_penalty

    def train_step(self, real_samples: tf.Tensor) -> typing.Dict[str, float]:
        batch_size = tf.shape(real_samples)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        gps = []

        # Step 1. Train the discriminator with both real and fake samples
        # Train the discriminator more often than the generator
        for _ in range(self.discriminator_extra_steps):

            # Step 1. Train the discriminator with both real images and fake images
            with tf.GradientTape() as tape:
                fake_samples = self.generator(noise, training=True)
                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)

                # Add instance noise to real and fake samples
                real_samples = self.add_instance_noise(real_samples)
                fake_samples = self.add_instance_noise(fake_samples)

                # Calculate the WGAN-GP gradient penalty
                gp = self.gradient_penalty(real_samples, fake_samples, self.discriminator)
                gps.append(gp)

                # Add gradient penalty to the original discriminator loss 
                disc_loss = self.discriminator_loss(pred_real, pred_fake) + gp * self.gp_weight 

            # Compute discriminator gradients
            grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update discriminator weights
            self.discriminator_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Step 2. Train the generator
        with tf.GradientTape() as tape:
            fake_samples = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator_loss(pred_fake)

        # Compute generator gradients
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Update generator wieghts
        self.generator_opt.apply_gradients(zip(grads, self.generator.trainable_variables))   

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(real_samples, fake_samples)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"d_loss": disc_loss, "g_loss": gen_loss, "gp": tf.reduce_mean(gps)})

        return results


class ResultsCallback(tf.keras.callbacks.Callback):
    """ Callback for generating and saving images during training."""
    def __init__(
            self, 
            noise_dim: int, 
            output_path: str, 
            examples_to_generate: int=16, 
            grid_size: tuple=(4, 4), 
            spacing: int=5, 
            gif_size: tuple=(416, 416), 
            duration: float=0.1, 
            save_model: bool=True
        ) -> None:
        super(ResultsCallback, self).__init__()
        self.seed = tf.random.normal([examples_to_generate, noise_dim])
        self.results = []
        self.output_path = output_path
        self.results_path = output_path + '/results'
        self.grid_size = grid_size
        self.spacing = spacing
        self.gif_size = gif_size
        self.duration = duration
        self.save_model = save_model

        os.makedirs(self.results_path, exist_ok=True)

    def save_plt(self, epoch: int, results: np.ndarray):
        # construct an image from generated images with spacing between them using numpy
        w, h , c = results[0].shape
        # construct grind with self.grid_size
        grid = np.zeros((self.grid_size[0] * w + (self.grid_size[0] - 1) * self.spacing, self.grid_size[1] * h + (self.grid_size[1] - 1) * self.spacing, c), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid[i * (w + self.spacing):i * (w + self.spacing) + w, j * (h + self.spacing):j * (h + self.spacing) + h] = results[i * self.grid_size[1] + j]

        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        # save the image
        cv2.imwrite(f'{self.results_path}/img_{epoch}.png', grid)

        # save image to memory resized to gif size
        self.results.append(cv2.resize(grid, self.gif_size, interpolation=cv2.INTER_AREA))

    def on_epoch_end(self, epoch: int, logs: dict=None):
        # Define your custom code here that should be executed at the end of each epoch
        predictions = self.model.generator(self.seed, training=False)
        predictions_uint8 = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
        self.save_plt(epoch, predictions_uint8)

        if self.save_model:
            # save keras model to disk
            models_path = os.path.join(self.output_path, "model")
            os.makedirs(models_path, exist_ok=True)
            self.model.discriminator.save(models_path + "/discriminator.h5")
            self.model.generator.save(models_path + "/generator.h5")

    def on_train_end(self, logs: dict=None):
        # save the results as a gif with imageio

        # Create a list of imageio image objects from the OpenCV images
        # image is in BGR format, convert to RGB format when loading
        imageio_images = [imageio.core.util.Image(image[...,::-1]) for image in self.results]

        # Write the imageio images to a GIF file
        imageio.mimsave(self.results_path + "/output.gif", imageio_images, duration=self.duration)


class LRSheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler for WGAN-GP"""
    def __init__(self, decay_epochs: int, tb_callback=None, min_lr: float=0.00001):
        super(LRSheduler, self).__init__()
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
        self.tb_callback = tb_callback
        self.compiled = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.compiled:
            self.generator_lr = self.model.generator_opt.lr.numpy()
            self.discriminator_lr = self.model.discriminator_opt.lr.numpy()
            self.compiled = True

        if epoch < self.decay_epochs:
            new_g_lr = max(self.generator_lr * (1 - (epoch / self.decay_epochs)), self.min_lr)
            self.model.generator_opt.lr.assign(new_g_lr)
            new_d_lr = max(self.discriminator_lr * (1 - (epoch / self.decay_epochs)), self.min_lr)
            self.model.discriminator_opt.lr.assign(new_d_lr)
            print(f"Learning rate generator: {new_g_lr}, discriminator: {new_d_lr}")

            # Log the learning rate on TensorBoard
            if self.tb_callback is not None:
                writer = self.tb_callback._writers.get('train')  # get the writer from the TensorBoard callback
                with writer.as_default():
                    tf.summary.scalar('generator_lr', data=new_g_lr, step=epoch)
                    tf.summary.scalar('discriminator_lr', data=new_d_lr, step=epoch)
                    writer.flush()

# celebA dataset path
dataset_path = "Dataset/img_align_celeba"

# Set the input shape and size for the generator and discriminator
batch_size = 128
img_shape = (64, 64, 3) # The shape of the input image, input to the discriminator
noise_dim = 128 # The dimension of the noise vector, input to the generator
model_path = 'Models/02_WGANGP_faces'
os.makedirs(model_path, exist_ok=True)

# Define your data generator
datagen = ImageDataGenerator(
    preprocessing_function=lambda x: (x / 127.5) - 1.0,  # Normalize image pixel values to [-1, 1]
    horizontal_flip=True  # Data augmentation
)

# Create a generator that yields batches of images
train_generator = datagen.flow_from_directory(
    directory=dataset_path,  # Path to directory containing images
    target_size=img_shape[:2],  # Size of images (height, width)
    batch_size=batch_size,
    class_mode=None,  # Do not use labels
    shuffle=True,  # Shuffle the data
)

generator = build_generator(noise_dim)
generator.summary()

discriminator = build_discriminator(img_shape)
discriminator.summary()

generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)

# Wasserstein loss for the discriminator
def discriminator_w_loss(pred_real, pred_fake):
    real_loss = tf.reduce_mean(pred_real)
    fake_loss = tf.reduce_mean(pred_fake)
    return fake_loss - real_loss

# Wasserstein loss for the generator
def generator_w_loss(pred_fake):
    return -tf.reduce_mean(pred_fake)

callback = ResultsCallback(noise_dim=noise_dim, output_path=model_path, duration=0.04)
tb_callback = TensorBoard(model_path + '/logs')
lr_scheduler = LRSheduler(decay_epochs=500, tb_callback=tb_callback, min_lr=0.00002)

gan = WGAN_GP(discriminator, generator, noise_dim, discriminator_extra_steps=5)
gan.compile(discriminator_optimizer, generator_optimizer, discriminator_w_loss, generator_w_loss, run_eagerly=False)

gan.fit(train_generator, epochs=500, callbacks=[callback, tb_callback, lr_scheduler])