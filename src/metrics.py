import keras
import numpy as np
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(keras.callbacks.Callback):
    """
    Implementation of F1, Precision, and Recall macro metrics according to:
    https://github.com/keras-team/keras/issues/5794
    """

    def on_epoch_end(self, batch, logs={}):
        print("\nEvaluating on the validation set")
        yp = self.model.predict(self.validation_data[0],
                                batch_size=self.params['batch_size'], verbose=0)

        predict = np.argmax(yp, axis=1)
        targ = np.argmax(self.validation_data[1], axis=1)

        f1 = f1_score(targ, predict, average="macro")
        prec = precision_score(targ, predict, average="macro")
        rec = recall_score(targ, predict, average="macro")

        print("F1: {:.4f}, Precision: {:.4f}, Recall {:.4f}\n".format(
            f1, prec, rec))


def top_3_acc(x, y, top_k=3):
    return top_k_categorical_accuracy(x, y, k=top_k)
