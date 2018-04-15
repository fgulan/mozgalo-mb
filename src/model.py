from keras.applications.densenet import DenseNet169
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import model_from_json

def XceptionModel(input_shape, num_classes, weights="imagenet", 
                  include_top=False, fine_tune=False):
    """
    Xception model.
    :param input_shape: Input shape
    :param num_classes: Number of classes
    :param weights: Pre-trained weights dataset. None if training from scratch.
    :param include_top: Include FC layers or not.
    :param fine_tune: If True then all layers will be trainable, otherwise only last layer will be trainable
    :return: predictions layer and the base_model variable
    """
    base_model = Xception(input_shape=input_shape, weights=weights, include_top=include_top)

    if not fine_tune:
        for layer in base_model.layers:
            layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    return predictions, base_model


def InceptionResentV2Model(input_shape, num_classes, weights="imagenet", include_top=False):
    """
    InceptionResnetV2 (InceptionV4) model.

    :param input_shape: Input shape
    :param num_classes: Number of classes
    :param weights: Pre-trained weights dataset. None if training from scratch.
    :param include_top: Include FC layers or not.
    :return: predictions layer and the base_model variable
    """
    base_model = InceptionResNetV2(input_shape=input_shape, weights=weights,
                                   include_top=include_top)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return predictions, base_model


def DenseNet169Model(input_shape, num_classes, weights="imagenet", include_top=False):
    """
    DenseNet169 model.

    :param input_shape: Input shape
    :param num_classes: Number of classes
    :param weights: Pre-trained weights dataset. None if training from scratch.
    :param include_top: Include FC layers or not.
    :return: predictions layer and the base_model variable
    """
    base_model = DenseNet169(input_shape=input_shape, weights=weights,
                             include_top=include_top)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return predictions, base_model

def CustomModel(model_path, weights_path, fine_tune=False):
    """
    Model from disk.

    :param model_path: Model JSON definition path
    :param weights_path: Weights file path
    :param fine_tune: If True then all layers will be trainable, otherwise only last layer will be trainable
    :param include_top: Include FC layers or not.
    :return: loaded model with weights
    """
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    
    for layer in model.layers:
        layer.trainable = fine_tune
    
    model.layers[-1].trainable = True

    return model