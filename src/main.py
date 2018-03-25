import os
from datetime import datetime

import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger, TensorBoard
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

import utils
from metrics import Metrics, top_3_acc
from model import XceptionModel


def get_callbacks(weights_file="models/weights.ep:{epoch:02d}-vloss:{val_loss:.4f}.hdf5",
                  save_epochs=1, patience=5, min_delta=0):
    """

    :param weights_file: string, path to save the model file.
    :param save_epochs: Interval (number of epochs) between checkpoints.
    :param patience: number of epochs with no improvement after which training will be stopped.
    :param min_delta: minimum change in the monitored quantity to qualify as an improvement,
     i.e. an absolute change of less than min_delta, will count as no improvement.
    :return: List of Keras callbacks
    """
    loggers = list()
    loggers.append(EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience,
                                 verbose=1))

    loggers.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=False,
                                   period=save_epochs))
    loggers.append(BaseLogger())
    loggers.append(Metrics())

    save_time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    loggers.append(TensorBoard(log_dir=os.path.join("logs", save_time)))

    return loggers


def main():
    # create the base pre-trained model
    # Keras will automatically download pre-trained weights if they are missing
    predictions, base_model = XceptionModel((299, 150, 3), 25)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # batch size 
    batch_size = 16

    # Create data generator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.3)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            '/Users/filipgulan/dataset-mozgalo',
            target_size=(299, 150),

            batch_size=batch_size,
            class_mode='categorical')

    optimizer = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           top_3_acc])

    # train the model on the new data for a few epochs
    model.fit_generator(
            train_generator,
            steps_per_epoch=1500 // batch_size,
            epochs=2, 
            validation_data=train_generator,
            validation_steps=500 // batch_size)


if __name__ == "__main__":
    # Set seeds for reproducible results
    utils.set_random_seeds()
    main()
