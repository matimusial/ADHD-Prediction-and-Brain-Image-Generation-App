from PyQt5.QtCore import QObject, pyqtSignal


class GenerateNewWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    model_loaded = pyqtSignal(object)

    def __init__(self, db, model_name):
        super().__init__()
        self.db = db
        self.model_name = model_name

    def run(self):
        try:
            model = self.db.select_model(self.model_name)
            self.model_loaded.emit(model)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
