from PyQt5.QtCore import QObject, pyqtSignal
from tensorflow.python.keras.callbacks import Callback


class TrainingManager:
    def __init__(self):
        self.stop_flag = False

    def stop_training(self):
        self.stop_flag = True

    def is_stopped(self):
        return self.stop_flag


class EarlyStoppingCallback(Callback):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        if self.manager.is_stopped():
            print("Stopped on epoch:", epoch)
            self.model.stop_training = True


class RealTimeMetricsCallback(QObject, Callback):
    metrics_updated = pyqtSignal(int, dict)

    def __init__(self, total_epochs):
        QObject.__init__(self)
        Callback.__init__(self)
        self.total_epochs = total_epochs
        self.metrics = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": []
        }

        self.gan_metrics = {
            "train_d_loss": [],
            "train_g_loss": [],
            "val_d_loss": [],
            "val_g_loss": [],
            "epoch_number": [],
            "img": None
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.metrics['accuracy'].append(logs.get('accuracy'))
        self.metrics['val_accuracy'].append(logs.get('val_accuracy'))
        self.metrics['loss'].append(logs.get('loss'))
        self.metrics['val_loss'].append(logs.get('val_loss'))

        self.metrics_updated.emit(epoch, self.metrics.copy())

    def update_gan_metrics(self, train_d_loss, train_g_loss, val_d_loss, val_g_loss, img, epoch):
        self.gan_metrics['train_d_loss'].append(train_d_loss)
        self.gan_metrics['train_g_loss'].append(train_g_loss)
        self.gan_metrics['val_d_loss'].append(val_d_loss)
        self.gan_metrics['val_g_loss'].append(val_g_loss)
        self.gan_metrics['epoch_number'].append(epoch)
        self.gan_metrics["img"] = img

        self.metrics_updated.emit(epoch, self.gan_metrics.copy())
