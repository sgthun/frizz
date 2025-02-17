import tensorflow as tf
import os
import sys
from models import create_model, evaluate_model, train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir, target_size=(150,150), batch_size=32):
    """
    Load images and labels from data_dir using TensorFlow's image_dataset_from_directory.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False  # For consistent evaluation ordering
    )
    return dataset

def main():
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Exiting.")
        sys.exit(1)

    print("Loading data from:", data_dir)
    test_dataset = load_data(data_dir)  # Now a tf.data.Dataset of images and labels

    # Create two example models with the correct number of classes (7)
    model_a = create_model(architecture="simple", num_classes=7)
    model_b = create_model(architecture="complex", num_classes=7)
    logger.info("Models created")
    
    # training steps:
    model_a, history_a = train_model(model_a, data_dir=data_dir)
    model_b, history_b = train_model(model_b, data_dir=data_dir)
    logger.info("Models trained")
    
    # Evaluate performance on the test dataset
    performance_a = evaluate_model(model_a, test_dataset)
    performance_b = evaluate_model(model_b, test_dataset)

    print("Model A performance:", performance_a)
    print("Model B performance:", performance_b)

    if performance_a > performance_b:
        print("Model A performed better!")
    elif performance_b > performance_a:
        print("Model B performed better!")
    else:
        print("Both models performed equally!")

if __name__ == "__main__":
    main()