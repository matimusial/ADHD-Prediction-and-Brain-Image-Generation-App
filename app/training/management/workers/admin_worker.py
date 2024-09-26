from PyQt5.QtCore import QObject, pyqtSignal


class AdminWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self):
        try:
            self.controller.train()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
