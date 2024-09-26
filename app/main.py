import os
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication

from app.controllers.admin.admin_db import AdminDbController
from app.controllers.admin.admin_mri_cnn import AdminMriCnnController
from app.controllers.doctor.doctor import DoctorController
from app.controllers.admin.admin_eeg_cnn import AdminEegCnnController
from app.controllers.doctor.generate_new_pic import GenerateNewController
from app.controllers.admin.admin_mri_gan import AdminMriGanController
from app.services.database import Database


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.admin_mri_gan_controller = None
        self.admin_mri_cnn_controller = None
        self.admin_db_controller = None
        self.doctor_controller = None
        self.generate_new_controller = None
        self.admin_eeg_cnn_controller = None

        self.database = Database()

        self._load_doctor_ui()
        self.show()

    def load_ui(self, path):
        try:
            return uic.loadUi(path, self)
        except Exception as e:
            print(f"Error: load_ui - {e}")

    def _load_gen_view(self):
        from app.properties.directory_config import DOCTOR_GEN_NEW_PIC_UI_PATH
        ui = self.load_ui(DOCTOR_GEN_NEW_PIC_UI_PATH)
        self.generate_new_controller = GenerateNewController(ui)

        ui.backBtn.clicked.connect(self._load_doctor_ui)

    def _load_doctor_ui(self):
        from app.properties.directory_config import DOCTOR_UI_PATH
        ui = self.load_ui(DOCTOR_UI_PATH)
        self.doctor_controller = DoctorController(self, ui)

        ui.switchSceneBtn.clicked.connect(self._load_admin_eeg_cnn)
        ui.generateNewBtn.clicked.connect(self._load_gen_view)

    def _load_admin_eeg_cnn(self):
        from app.properties.directory_config import ADMIN_EEG_CNN_UI_PATH
        ui = self.load_ui(ADMIN_EEG_CNN_UI_PATH)
        try:
            self.admin_eeg_cnn_controller = AdminEegCnnController(ui)
        except Exception as e:
            print(e)

        ui.switchSceneBtn.clicked.connect(self._load_doctor_ui)
        ui.cnnMriBtn.clicked.connect(self._load_admin_mri_cnn)
        ui.ganMriBtn.clicked.connect(self._load_admin_mri_gan)
        ui.dbAdminBtn.clicked.connect(self._load_admin_db_view)

    def _load_admin_db_view(self):
        from app.properties.directory_config import ADMIN_DB_UI_PATH
        ui = self.load_ui(ADMIN_DB_UI_PATH)
        self.admin_db_controller = AdminDbController(ui)

        ui.backBtn.clicked.connect(self._load_admin_eeg_cnn)

    def _load_admin_mri_cnn(self):
        from app.properties.directory_config import ADMIN_MRI_CNN_UI_PATH
        ui = self.load_ui(ADMIN_MRI_CNN_UI_PATH)
        self.admin_mri_cnn_controller = AdminMriCnnController(ui)

        ui.cnnEegBtn.clicked.connect(self._load_admin_eeg_cnn)
        ui.ganMriBtn.clicked.connect(self._load_admin_mri_gan)
        ui.dbAdminBtn.clicked.connect(self._load_admin_db_view)

    def _load_admin_mri_gan(self):
        from app.properties.directory_config import ADMIN_MRI_GAN_UI_PATH
        ui = self.load_ui(ADMIN_MRI_GAN_UI_PATH)
        self.admin_mri_gan_controller = AdminMriGanController(ui)

        ui.switchSceneBtn.clicked.connect(self._load_doctor_ui)
        ui.cnnEegBtn.clicked.connect(self._load_admin_eeg_cnn)
        ui.cnnMriBtn.clicked.connect(self._load_admin_mri_cnn)
        ui.dbAdminBtn.clicked.connect(self._load_admin_db_view)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
