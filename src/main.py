import os
import argparse
from datetime import datetime

import pdb
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger, TensorBoard
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from cv2 import GaussianBlur
from keras.metrics import categorical_accuracy
import numpy as np
import cv2

import utils
from metrics import top_3_acc
from model import XceptionModel, CustomModel

DATASET_ROOT_PATH = '/Users/filipgulan/college/mb-dataset1/'

def normalize(image):
    image /= 255.
    image -= 0.5

    return image

def resize(image, min_height = 550):
    """
    Resize the image so the maximum height is min_height
    """
    h, w, _ = image.shape

    new_h = min(min_height, h)
    new_w = int(new_h * w / h)

    return cv2.resize(image, (new_w, new_h))

def crop_upper_part(image, percent=0.4):
    height, _, _ = image.shape
    point = int(percent * height)
    return image[:point, :]

def random_crop(value, size):
    """
    Performs the random crop of the patch size size
    """
    h, w, _ = value.shape
    top_x = np.random.randint(0, w - size[1])
    top_y = np.random.randint(0, h - size[0])

    return value[top_y:h+top_y, top_x: top_x+w, :]

def random_erase(value):
    """
    Performs random erasing augmentation technique.
    https://arxiv.org/pdf/1708.04896.pdf
    """
    h, w, _ = value.shape

    r_width = np.random.randint(20, w - 20)
    r_height = np.random.randint(20, h - 20)

    top_x = np.random.randint(0, w - r_width)
    top_y = np.random.randint(0, h - r_height)

    value[top_y:r_height+top_y, top_x:top_x+r_width, :] = np.mean(value)

    return value

def random_gauss_blur(image):
    return GaussianBlur(image, (3, 3), 0)


def preprocess_image_train(image):
    img_array = img_to_array(image).astype(np.float32)
    #img_array = resize(img_array)
    img_array = crop_upper_part(img_array, 0.5)

    # if np.random.random() < 0.5:
    #    img_array = random_crop(img_array, size=(299, 164))

    if np.random.random() < 0.5:
        img_array = random_gauss_blur(img_array)

    img_array = random_erase(img_array)
    img_array = normalize(img_array)

    return array_to_img(img_array, scale=False)

def preprocess_image_val(image):
    img_array = img_to_array(image).astype(np.float32)
    #img_array = resize(img_array)
    img_array = crop_upper_part(img_array, 0.5)

    img_array = normalize(img_array)

    return array_to_img(img_array, scale=False)

def get_callbacks(weights_file=os.path.join("models", "weights_ep{epoch:02d}.hd5f"),
                  save_epochs=1, patience=20, min_delta=0):
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
                                   save_best_only=True, save_weights_only=True,
                                   period=save_epochs))
    loggers.append(BaseLogger())

    # Validation data is not available when using flow_from_directory
    # loggers.append(Metrics())

    save_time = datetime.now().strftime('%d%m%Y%H%M%S')
    loggers.append(TensorBoard(log_dir=os.path.join("logs", save_time)))

    return loggers

def train(args):
    # Paramaters
    num_classes = 25
    batch_size = 2
    num_channels = 3
    input_size = (400, 400)  # h x w
    epochs = 40
    learning_rate = 0.0005

    # Use less data for faster experimenting
    sample = 1.0
    assert 0.0 < sample <= 1

    # Number of feed workers (should be equal to the number of virtual CPU threads)
    workers = 8

    if args.model != None and args.weights != None:
        model = CustomModel(args.model, args.weights, fine_tune=args.fine_tune)
    else:
        predictions, base_model = XceptionModel((*input_size, num_channels), 
                                                num_classes, fine_tune=args.fine_tune)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

    # Create data generators
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_train,
                                       samplewise_center=False,
                                       samplewise_std_normalization=False,
                                       rotation_range=9,
                                       zoom_range=(0.4, 1.2),
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_val,
                                            samplewise_center=False,
                                            samplewise_std_normalization=False)

    # this is a generator that will and indefinitely
    # generate batches of augmented image data
    # target_size: tuple of integers (height, width)
    train_flow = train_datagen.flow_from_directory(
        os.path.join(DATASET_ROOT_PATH, 'train'),
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_flow = validation_datagen.flow_from_directory(
        os.path.join(DATASET_ROOT_PATH, 'validation'),
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')

    # Learning rate at the end of training
    final_lr = 2e-5
    decay = ((1 / (final_lr / learning_rate)) - 1) / (
    round((sample * train_flow.samples)) // batch_size * epochs)

    optimizer = optimizers.RMSprop(lr=learning_rate,
                                   decay=decay)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[categorical_accuracy, top_3_acc])

    # Store model configuration
    model_json = model.to_json()
    with open(os.path.join("models", "model.json"), "w") as json_file:
        json_file.write(model_json)

    # train the model on the new data for a few epochs
    model.fit_generator(
        train_flow,
        workers=workers,
        max_queue_size=round(workers * 1.7),  # tweak if needed
        use_multiprocessing=False,
        steps_per_epoch=round((sample * train_flow.samples)) // batch_size,
        epochs=epochs,
        callbacks=get_callbacks(patience=7),
        shuffle=True,
        validation_data=validation_flow,
        validation_steps=round((sample * validation_flow.samples)) // batch_size)


def main():
    # Set seeds for reproducible results
    utils.set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-f', '--fine-tune', type=utils.str2bool, default=False)
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()