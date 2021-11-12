import numpy as np

from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    """

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)
