# import numpy as np
# import tensorflow as tf
# from generator import create_generator
# from discriminator import create_discriminator
# from gan import create_gan
# from data_preprocessing import load_data

# # Load data
# dataset = load_data()

# # Initialize generator and discriminator
# generator = create_generator()
# discriminator = create_discriminator()

# # Create the GAN model
# gan = create_gan(generator, discriminator)

# # Compile models
# discriminator.compile(optimizer='adam', loss='binary_crossentropy')
# gan.compile(optimizer='adam', loss='binary_crossentropy')

# # Training loop
# epochs = 10000
# batch_size = 32

# for epoch in range(epochs):
#     # Get a batch of real images
#     real_images = next(iter(dataset))  # Assuming 'dataset' is already loaded

#     # Generate random noise
#     noise = np.random.normal(0, 1, (batch_size, 100))

#     # Generate fake images
#     fake_images = generator.predict(noise)

#     # Labels for real and fake images
#     real_labels = np.ones((batch_size, 1))  # Real image labels
#     fake_labels = np.zeros((batch_size, 1))  # Fake image labels

#     # Train discriminator
#     discriminator.train_on_batch(real_images, real_labels)
#     discriminator.train_on_batch(fake_images, fake_labels)

#     # Train generator (via GAN)
#     gan.train_on_batch(noise, real_labels)  # Generator tries to fool the discriminator

#     # Every few epochs, you can visualize or save generated images here

# # After training, generate and visualize an image
# def generate_image():
#     noise = np.random.normal(0, 1, (1, 100))  # Random noise for input
#     generated_image = generator.predict(noise)
#     return generated_image

# import matplotlib.pyplot as plt

# generated_img = generate_image()
# plt.imshow((generated_img[0] + 1) / 2)  # Rescale to [0, 1] for display
# plt.show()
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from generator import create_generator
from discriminator import create_discriminator
from gan import create_gan
from data_preprocessing import load_data  # Import the load_data function

# Function to generate a new image from the trained generator
def generate_image(generator):
    noise = np.random.normal(0, 1, (1, 100))  # Generate random noise
    generated_image = generator.predict(noise)
    return generated_image

# Function to save generated images
def save_generated_image(generated_img, epoch):
    # Rescale generated image to [0, 1]
    generated_img_rescaled = (generated_img[0] + 1) / 2.0
    plt.imshow(generated_img_rescaled)
    plt.axis('off')  # Hide axes
    plt.savefig(f"generated_image_epoch_{epoch}.png")
    plt.close()

# Function to visualize and display generated images
def visualize_image(image):
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

# Load data (using a smaller batch size for quicker iterations)
dataset = load_data(image_size=(64, 64))  # Ensure to load the correct size dataset

# Initialize generator and discriminator models
generator = create_generator()
discriminator = create_discriminator()

# Create the GAN model
gan = create_gan(generator, discriminator)

# Compile models with adjusted learning rates to speed up training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

# Parameters
epochs = 1000  # Reduce the number of epochs for testing/debugging
batch_size = 16  # Reduced batch size to speed up training
sample_interval = 100  # Interval to save/generated images during training

# Training loop
for epoch in range(epochs):
    for real_images in dataset:
        # Ensure the batch size of real_images is the same as the fake labels and real labels
        batch_size = real_images.shape[0]  # Dynamically set batch size from the data

        # Labels for real and fake images (now match the batch size of real_images)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, 100))  # Generate random noise
        fake_images = generator.predict(noise)  # Generate fake images

        # Train the discriminator on real images
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)

        # Train the discriminator on fake images
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # Train the generator (via the GAN model)
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)[0]}] [G loss: {g_loss}]")

        # Save and display generated images periodically
        if epoch % sample_interval == 0:
            # Generate and save images
            generated_img = generate_image(generator)
            save_generated_image(generated_img, epoch)
            visualize_image(generated_img[0])  # Display the generated image
            break  # Exit the for-loop for the current epoch to prevent excessive computation
