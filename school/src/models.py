import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(architecture="simple", input_shape=(150,150,3), num_classes=10):
    """
    Create and return a TensorFlow model based on the chosen architecture.
    """
    if architecture == "simple":
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    else:
        # Option for a more complex architecture.
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, data_dir='/app/data', batch_size=8, epochs=2, target_size=(150,150)):
    """
    Train the provided TensorFlow model using training data in data_dir.
    
    The data is assumed to be organized in subdirectories (one per label), and each 
    <label>/image_<number>/ contains a .jpg file. This function uses ImageDataGenerator 
    to split the images into training (80%) and validation (20%) sets.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    return model, history

def evaluate_model(model, dataset):
    """
    Evaluate the trained TensorFlow model on test images and labels from a dataset.

    Args:
        model: A trained TensorFlow/Keras model.
        dataset: A tf.data.Dataset of (images, labels).

    Returns:
        test_accuracy: The accuracy obtained on the test set.
    """
    results = model.evaluate(dataset, verbose=0)
    test_accuracy = results[1]  # Assuming accuracy is the second metric.
    return test_accuracy