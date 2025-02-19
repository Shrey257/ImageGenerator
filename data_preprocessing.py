# import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory

# def load_data(image_size=(64, 64)):
#     # Replace 'path_to_your_image_folder' with the actual folder path containing images
#     dataset = image_dataset_from_directory(
#         'path_to_your_image_folder',
#         image_size=image_size,          # Resize images to (64, 64)
#         batch_size=32,                  # Batch size
#         label_mode=None,                # No labels needed for GAN
#         color_mode='rgb'                # For RGB images
#     )

#     # Normalize the images to [-1, 1] range for GAN
#     dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
#     return dataset


import tensorflow as tf
import tensorflow_datasets as tfds

def load_data(image_size=(64, 64)):
    # Load the CIFAR-10 dataset directly from TensorFlow Datasets
    dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)

    # Convert the dataset into a format compatible with the generator
    def preprocess_image(image, label):
        image = tf.image.resize(image, image_size)  # Resize the image
        image = (image - 127.5) / 127.5  # Normalize the image to [-1, 1]
        return image

    train_data = dataset['train'].map(preprocess_image).batch(32)

    return train_data
