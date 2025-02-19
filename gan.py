from tensorflow.keras import models

def create_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator for GAN training
    gan = models.Sequential([generator, discriminator])
    return gan
