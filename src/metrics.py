import keras
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(keras.callbacks.Callback):
    """
    Implementation of F1, Precision, and Recall macro metrics according to:
    https://github.com/keras-team/keras/issues/5794
    """

    def on_epoch_end(self, batch, logs={}):
        mbatch = self.params['batch_size']
        f1s, precs, recs = 0, 0, 0

        batches = len(self.validation_data[0]) // mbatch
        iters = batches if len(self.validation_data[0]) % mbatch == 0 else batches + 1

        print("Calculating metrics on the Validation data")

        for i in range(iters):
            batch_input = self.validation_data[0][i * mbatch:(i + 1) * mbatch]
            batch_output = self.validation_data[1][i * mbatch:(i + 1) * mbatch]

            predict = np.argmax(np.asarray(self.model.predict(batch_input)),
                                axis=1)
            targ = np.argmax(batch_output, axis=1)

            f1s += f1_score(targ, predict, average="macro")
            precs += precision_score(targ, predict, average="macro")
            recs += recall_score(targ, predict, average="macro")

        print("F1: {:.2f}, Precision: {:.2f}, Recall {:.2f}\n".format(
            f1s / iters, precs / iters, recs / iters))

        return
