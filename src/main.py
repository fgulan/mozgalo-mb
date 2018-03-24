import os
from datetime import datetime

import numpy as np
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.metrics import fbeta_score, precision, recall


def get_callbacks(weights_file="models/weights.{epoch:02d}-{val_loss:.3f}.hdf5",
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
                                   save_best_only=False, save_weights_only=False,
                                   period=save_epochs))
    loggers.append(BaseLogger())

    save_time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    loggers.append(TensorBoard(log_dir=os.path.join("logs", save_time)))

    return loggers


def main():
    # create the base pre-trained model
    # Keras will automatically download pre-trained weights if they are missing
    base_model = InceptionV3(input_shape=(299, 150, 3), weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    predictions = Dense(26, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # DUMMY DATA
    data = np.random.random((100, 299, 150, 3))
    labels = np.random.randint(26, size=(100, 1))

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

    # Convert labels to categorical one-hot encoding
    one_hot_train = to_categorical(y_train, num_classes=26)
    one_hot_val = to_categorical(y_val, num_classes=26)

    # compile the model (should be done *after* setting layers to non-trainable)
    optimizer = optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[fbeta_score, precision, recall])

    # train the model on the new data for a few epochs
    model.fit(x=x_train, y=one_hot_train,
              callbacks=get_callbacks(save_epochs=2, patience=4),
              batch_size=4, epochs=2, shuffle=True,
              validation_data=(x_val, one_hot_val))


if __name__ == "__main__":
    # Set seed
    np.random.seed(1337)

    main()
