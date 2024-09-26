from PyQt5.QtCore import QObject, pyqtSignal


class DoctorWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self):
        try:
            self.load_models()
            self.process_files()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def load_models(self):
        self.controller.load_models()

    def process_files(self):
        self.controller.process_files()
