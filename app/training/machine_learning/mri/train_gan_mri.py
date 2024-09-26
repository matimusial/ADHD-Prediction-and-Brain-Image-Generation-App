import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU, Input,
    BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam

from app.training.management.training_callbacks import RealTimeMetricsCallback
from app.properties.directory_config import TEMP_MODEL_PATH
from app.training.management.training_signal_functions import connect_signals_gan
from app.services.data_processing.mri_processing import normalize_mri, trim_mri
from app.utils.file_utils import read_pickle
from app.utils.plotting.mri_plot_utils import show_plot_mri


def build_generator():
    from app.properties.mri_config import GAN_INPUT_SHAPE_MRI
    model = Sequential()
    model.add(Input(shape=(100,)))
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Dense(120 * 120 * 1, activation='tanh'))
    model.add(Reshape(GAN_INPUT_SHAPE_MRI))
    return model


def build_discriminator():
    from app.properties.mri_config import GAN_INPUT_SHAPE_MRI
    model = Sequential()
    model.add(Input(shape=GAN_INPUT_SHAPE_MRI))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


def generate_image(generator, epoch):
    noise = np.random.normal(0, 1, [1, 100])
    generated_image = generator.predict(noise)
    generated_image = generated_image * 0.5 + 0.5
    qpm = show_plot_mri(generated_image[0], f'Epoch: {epoch}')
    return qpm


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(generator, discriminator, real_images, generator_optimizer, discriminator_optimizer):
    """
    Performs a single training step for the GAN, updating both the generator and discriminator.

    Returns:
    - d_loss: Discriminator loss after the step.
    - g_loss: Generator loss after the step.
    """
    from app.properties.mri_config import GAN_BATCH_SIZE_MRI

    noise = tf.random.normal([GAN_BATCH_SIZE_MRI, 100])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        d_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    noise = tf.random.normal([GAN_BATCH_SIZE_MRI, 100])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=False)

        g_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, g_loss


@tf.function
def val_step(generator, discriminator, real_images):
    from app.properties.mri_config import GAN_BATCH_SIZE_MRI

    noise = tf.random.normal([GAN_BATCH_SIZE_MRI, 100])
    generated_images = generator(noise, training=False)

    real_output = discriminator(real_images, training=False)
    fake_output = discriminator(generated_images, training=False)

    d_loss = discriminator_loss(real_output, fake_output)
    g_loss = generator_loss(fake_output)

    return d_loss, g_loss


def train_gan(data_type, manager_instance, controller_instance):
    """
    Trains a GAN for MRI data, managing the generator and discriminator and reporting metrics.
    """
    from app.properties.mri_config import (
        GAN_EPOCHS_MRI, GAN_LEARNING_RATE, INFO_GAN_DISP_INTERVAL,
        TEST_SIZE_MRI_GAN, GAN_BATCH_SIZE_MRI, GAN_INPUT_SHAPE_MRI)
    from app.properties.directory_config import ADHD_MRI_REAL_PICKLE_PATH, CONTROL_MRI_REAL_PICKLE_PATH

    print(f"Gan training for {data_type} started for {GAN_EPOCHS_MRI} epochs...")
    print(f"Results will be displayed every {INFO_GAN_DISP_INTERVAL}")
    print("\n")

    if data_type == "adhd":
        data = read_pickle(ADHD_MRI_REAL_PICKLE_PATH)
    elif data_type == "control":
        data = read_pickle(CONTROL_MRI_REAL_PICKLE_PATH)

    trimmed = trim_mri(data, GAN_INPUT_SHAPE_MRI[:2])
    normalized = normalize_mri(trimmed)

    train_data, val_data = train_test_split(normalized, test_size=TEST_SIZE_MRI_GAN)
    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)

    def fit_function():
        """
        Returns:
        - The average generator loss over the epochs.
        """
        discriminator_optimizer = Adam(GAN_LEARNING_RATE)
        generator_optimizer = Adam(GAN_LEARNING_RATE)
        generator = build_generator()
        discriminator = build_discriminator()

        real_time_metrics = RealTimeMetricsCallback(total_epochs=GAN_EPOCHS_MRI)
        connect_signals_gan(controller_instance, real_time_metrics)

        gen_loss_list = []

        img = show_plot_mri(np.zeros(GAN_INPUT_SHAPE_MRI), "")

        real_time_metrics.update_gan_metrics(0, 0, 0, 0, img, 0)

        for epoch in range(GAN_EPOCHS_MRI):
            if manager_instance.is_stopped():
                if len(gen_loss_list) >= 1:
                    return sum(gen_loss_list) / len(gen_loss_list)
                else:
                    return 0

            idx = np.random.randint(0, train_data.shape[0], GAN_BATCH_SIZE_MRI)
            real_imgs = train_data[idx]

            d_loss, g_loss = train_step(generator, discriminator, real_imgs, generator_optimizer,
                                        discriminator_optimizer)

            if (epoch + 1) % (GAN_EPOCHS_MRI // 20) == 0:
                gen_loss_list.append(g_loss.numpy())

            if (epoch + 1) % INFO_GAN_DISP_INTERVAL == 0:
                val_idx = np.random.randint(0, val_data.shape[0], GAN_BATCH_SIZE_MRI)
                val_real_imgs = val_data[val_idx]
                val_d_loss, val_g_loss = val_step(generator, discriminator, val_real_imgs)

                img = generate_image(generator, epoch + 1)
                real_time_metrics.update_gan_metrics(d_loss.numpy(), g_loss.numpy(),
                                                     val_d_loss.numpy(), val_g_loss.numpy(), img, epoch)

                print(
                    f"Epoch {epoch + 1} [Train D loss: {d_loss.numpy():.4f} | "
                    f"Train G loss: {g_loss.numpy():.4f}]")
                print(
                    f"Epoch {epoch + 1} [Val D loss: {val_d_loss.numpy():.4f} | "
                    f"Val G loss: {val_g_loss.numpy():.4f}]")

        generator.save(os.path.join(TEMP_MODEL_PATH, f'{round(sum(gen_loss_list) / len(gen_loss_list), 4)}.keras'))
        return sum(gen_loss_list) / len(gen_loss_list)

    gen_loss = fit_function()

    print(f"Generator loss: {round(gen_loss, 4)}")
    return round(gen_loss, 4)
