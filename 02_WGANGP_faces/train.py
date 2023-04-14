import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
from tqdm import tqdm

from keras.callbacks import TensorBoard

from model import build_generator, build_discriminator

class WGAN_GP(tf.keras.models.Model):
    def __init__(self, discriminator, generator, noise_dim, discriminator_extra_steps, gp_weight=10.0):
        super(WGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, discriminator_opt, generator_opt, discriminator_loss, generator_loss, **kwargs):
        super(WGAN_GP, self).compile(**kwargs)
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        Gradient penalty is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Generate random values for alpha
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

        # 1. Interpolate between real and fake images
        interpolated_images = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            # 2. Get the Critic's output for the interpolated image
            pred = self.discriminator(interpolated_images, training=True)

        # 3. Calculate the gradients w.r.t to the interpolated image
        gradients = gp_tape.gradient(pred, interpolated_images)

        # 4. Calculate the norm of the gradients.
        norm = tf.norm(tf.reshape(gradients, [batch_size, -1]), axis=1)
        # 5. Calculate gradient penalty
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)

        return gradient_penalty

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        # Step 1. Train the discriminator with both real images and fake images
        # Train the critic more often than the generator by 5 times (self.c_extra_steps) 
        for _ in range(self.discriminator_extra_steps):

            # Step 1. Train the discriminator with both real images and fake images
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)
                pred_real = self.discriminator(real_images, training=True)
                pred_fake = self.discriminator(fake_images, training=True)

                # Calculate the WGAN-GP gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                # Add gradient penalty to the original discriminator loss 
                disc_loss = self.discriminator_loss(pred_real, pred_fake) + gp * self.gp_weight 

            # Compute discriminator gradients
            grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update discriminator weights
            self.discriminator_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Step 2. Train the generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_images, training=True)
            gen_loss = self.generator_loss(pred_fake)

        # Compute generator gradients
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)

        # Update generator wieghts
        self.generator_opt.apply_gradients(zip(grads, self.generator.trainable_variables))   

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(real_images, fake_images)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"d_loss": disc_loss, "g_loss": gen_loss})

        return results


class ResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self, noise_dim, output_path, examples_to_generate=16, grid_size=(4, 4), spacing=5, gif_size=(416, 416), duration=0.1):
        super(ResultsCallback, self).__init__()
        self.seed = tf.random.normal([examples_to_generate, noise_dim])
        self.results = []
        self.output_path = output_path
        self.results_path = output_path + '/results'
        self.grid_size = grid_size
        self.spacing = spacing
        self.gif_size = gif_size
        self.duration = duration

        os.makedirs(self.results_path, exist_ok=True)

    def save_plt(self, epoch, results):
        # construct an image from generated images with spacing between them using numpy
        w, h , c = results[0].shape
        # construct grind with self.grid_size
        grid = np.zeros((self.grid_size[0] * w + (self.grid_size[0] - 1) * self.spacing, self.grid_size[1] * h + (self.grid_size[1] - 1) * self.spacing, c), dtype=np.uint8)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid[i * (w + self.spacing):i * (w + self.spacing) + w, j * (h + self.spacing):j * (h + self.spacing) + h] = results[i * self.grid_size[1] + j]

        # save the image
        cv2.imwrite(f'{self.results_path}/img_{epoch}.png', grid)

        # save image to memory resized to gif size
        self.results.append(cv2.resize(grid, self.gif_size, interpolation=cv2.INTER_AREA))

    def on_epoch_end(self, epoch, logs=None, save_model=True):
        # Define your custom code here that should be executed at the end of each epoch
        predictions = self.model.generator(self.seed, training=False)
        predictions_uint8 = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
        self.save_plt(epoch, predictions_uint8)

        if save_model:
            # save keras model to disk
            models_path = os.path.join(self.output_path, "model")
            os.makedirs(models_path, exist_ok=True)
            self.model.discriminator.save(models_path + "/discriminator.h5")
            self.model.generator.save(models_path + "/generator.h5")

    def on_train_end(self, logs=None):
        # save the results as a gif with imageio

        # Create a list of imageio image objects from the OpenCV images
        # image is in BGR format, convert to RGB format when loading
        imageio_images = [imageio.core.util.Image(image[...,::-1]) for image in self.results]

        # Write the imageio images to a GIF file
        imageio.mimsave(self.results_path + "/output.gif", imageio_images, duration=self.duration)


# face dataset path
dataset_path = "Dataset/img_align_celeba/img_align_celeba"

# Set the input shape and size for the generator and discriminator
img_shape = (64, 64, 3) # The shape of the input image, input to the discriminator
noise_dim = 128 # The dimension of the noise vector, input to the generator
model_path = 'Models/02_WGANGP_faces'
os.makedirs(model_path, exist_ok=True)

train_images = []
for image_path in tqdm(os.listdir(dataset_path)):
    img = cv2.imread(os.path.join(dataset_path, image_path))
    img = cv2.resize(img, (img_shape[0], img_shape[1]))
    train_images.append(img)

    # limit dataset to 100.000
    # if len(train_images) >= 10000:
    #     break

# Normalize image pixel values to [-1, 1]
train_images = (np.array(train_images).astype('float32') - 127.5) / 127.5 

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

callback = ResultsCallback(noise_dim=noise_dim, output_path=model_path)
tb_callback = TensorBoard(model_path + '/logs', update_freq=1)

gan = WGAN_GP(discriminator, generator, noise_dim, discriminator_extra_steps=5)
gan.compile(discriminator_optimizer, generator_optimizer, discriminator_w_loss, generator_w_loss, run_eagerly=False)

gan.fit(train_images, epochs=200, batch_size=128, callbacks=[callback, tb_callback])