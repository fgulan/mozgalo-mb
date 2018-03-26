from keras.applications.densenet import DenseNet169
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D


def XceptionModel(input_shape, num_classes, weights="imagenet", include_top=False):
    """
    Xception model.
    :param input_shape: Input shape
    :param num_classes: Number of classes
    :param weights: Pre-trained weights dataset. None if training from scratch.
    :param include_top: Include FC layers or not.
    :return: predictions layer and the base_model variable
    """
    base_model = Xception(input_shape=input_shape, weights=weights, include_top=include_top)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

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
