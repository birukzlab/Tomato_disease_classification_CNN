''''
import os
import sys
from cnn.exception import CustomException
from cnn import logging
import tensorflow as tf
from keras import preprocessing
from keras import models, layers
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_dir: str = os.path.join("research/data_tomato")  # Path to the directory with all the data
    train_data_path: str = os.path.join('artifacts', 'data_ingestion', 'train')
    val_data_path: str = os.path.join('artifacts', 'data_ingestion', 'val')
    test_data_path: str = os.path.join('artifacts', 'data_ingestion', 'test')
    image_size: tuple = (256, 256)  # Assuming IMAGE_SIZE is defined elsewhere
    batch_size: int = 32  # Assuming BATCH_SIZE is defined elsewhere
    seed: int = 123  # Seed for shuffling

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_and_split_dataset(self):
        logging.info("Loading and splitting dataset.")
        
        # Load dataset from the directory
        dataset = preprocessing.image_dataset_from_directory(
            self.config.data_dir,
            shuffle=True,
            seed=self.config.seed,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size
        )
        
        # Calculate size of the splits
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        train_size = int(0.85 * dataset_size)
        val_size = int(0.10 * dataset_size)
        test_size = dataset_size - (train_size + val_size)
        
        # Split the dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)


        
        # Perform prefetch and caching optimizations
        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # If you need to save the splits to the filesystem, that would involve additional logic here.
        # It is not typically done as `tf.data.Dataset` is meant for in-memory use.


        
        return train_dataset, val_dataset, test_dataset

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion.")
        self.load_and_split_dataset()
'''
import os
import tensorflow as tf
from keras.preprocessing.image import save_img
from keras.utils import image_dataset_from_directory
from dataclasses import dataclass
from cnn import logging  # Assuming cnn.logging is configured properly

@dataclass
class DataIngestionConfig:
    data_dir: str = os.path.join('research/data_tomato')
    output_dir: str = os.path.join('artifacts', 'data_ingestion')
    image_size: tuple = (256, 256)
    batch_size: int = 32
    seed: int = 123


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        # Ensure output directories exist
        self.train_dir = os.path.join(config.output_dir, 'train')
        self.val_dir = os.path.join(config.output_dir, 'val')
        self.test_dir = os.path.join(config.output_dir, 'test')
        for path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(path, exist_ok=True)

    def load_dataset(self):
        return image_dataset_from_directory(
            self.config.data_dir,
            shuffle=True,
            seed=self.config.seed,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size
        )

    def get_dataset_partitions_tf(self, dataset, train_split=0.85, val_split=0.1, test_split=0.05, shuffle=True, shuffle_size=10000):
        assert train_split + val_split + test_split == 1, "Splits must sum to 1"

        dataset_size = dataset.cardinality().numpy()
        if shuffle:
            dataset = dataset.shuffle(shuffle_size, seed=self.config.seed)

        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)

        return train_dataset, val_dataset, test_dataset

    def save_dataset_split(self, dataset, save_path, class_names):
        for images, labels in dataset.unbatch().batch(1):
            for image, label in zip(images, labels):
                # Convert tensor to image
                image = tf.cast(image, tf.uint8).numpy()
                label = label.numpy()
                # Determine class and filename
                class_dir = os.path.join(save_path, class_names[label])
                os.makedirs(class_dir, exist_ok=True)
                filename = f'{label}_{tf.timestamp()}.png'
                file_path = os.path.join(class_dir, filename)
                save_img(file_path, image)

    def initiate_data_ingestion_and_save(self):
        dataset = self.load_dataset()
        class_names = dataset.class_names

        train_dataset, val_dataset, test_dataset = self.get_dataset_partitions_tf(dataset)

        self.save_dataset_split(train_dataset, self.train_dir, class_names)
        self.save_dataset_split(val_dataset, self.val_dir, class_names)
        self.save_dataset_split(test_dataset, self.test_dir, class_names)
        logging.info("Data ingestion and saving complete.")
