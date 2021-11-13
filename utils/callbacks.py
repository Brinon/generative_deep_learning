import os

import numpy as np

from tensorflow.keras.callbacks import LearningRateScheduler, Callback


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)


class ModelSaver(Callback):
    def __init__(self, model_folder: str, model_name: str):
        self.model_folder = model_folder
        self.model_name = model_name

    def on_train_begin(self, logs=None):
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        if any(x.startswith(self.model_name) for x in os.listdir(self.model_folder)):
            raise ValueError(
                f"Already exists models called {self.model_name} in {self.model_folder}"
            )

    def on_epoch_end(self, epoch, logs=None):
        model_fpath = f"{self.model_folder}/{self.model_name}_{epoch:03d}.h5"
        self.model.save(model_fpath, overwrite=True)
