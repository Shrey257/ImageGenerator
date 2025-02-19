from tensorflow.keras import layers, models

def create_discriminator():
    discriminator = models.Sequential()
    discriminator.add(layers.Flatten(input_shape=(64, 64, 3)))  # Flatten the image
    discriminator.add(layers.Dense(512, activation='relu'))
    discriminator.add(layers.Dense(256, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))  # Output real/fake (binary)
    return discriminator
