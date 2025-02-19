from tensorflow.keras import layers, models

def create_generator():
    generator = models.Sequential()
    generator.add(layers.Dense(256, input_dim=100, activation='relu'))
    generator.add(layers.Dense(512, activation='relu'))
    generator.add(layers.Dense(1024, activation='relu'))
    generator.add(layers.Dense(64*64*3, activation='tanh'))
    generator.add(layers.Reshape((64, 64, 3)))  # Reshaped to (64, 64, 3) for RGB images
    return generator
