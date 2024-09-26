from PyQt5.QtGui import QPixmap

from app.training.machine_learning.mri.train_gan_mri import train_gan
from app.services.database import Database
from app.services.gui_services.general import on_exit
from app.services.gui_services.training_gui import get_main_train_ui_values, on_finished
from app.services.gui_services.alerts import show_warning_alert
from app.utils.gui_utils import set_enabled_buttons_boolean
from app.utils.file_utils import clear_temp_model_directory
from app.services.database_services import send_to_db
from app.training.management.create_thread_functions import create_admin_controller_thread
from app.training.management.training_callbacks import TrainingManager
from app.utils.file_utils import read_pickle
from app.training.management.training_signal_functions import update_gui_gan


class AdminMriGanController:
    def __init__(self, ui):

        from app.properties.directory_config import (ADHD_MRI_REAL_PICKLE_PATH, CONTROL_MRI_REAL_PICKLE_PATH,
                                                     MAGNIFYING_GLASS_BRAIN_ICON_PATH)

        self.ui = ui
        self.db_conn = Database()

        adhd_data = read_pickle(ADHD_MRI_REAL_PICKLE_PATH)
        control_data = read_pickle(CONTROL_MRI_REAL_PICKLE_PATH)
        self.loaded_adhd_files = len(adhd_data)
        self.loaded_control_files = len(control_data)
        self.current_channels = len(adhd_data[0])
        del adhd_data, control_data

        self.training_manager = TrainingManager()

        self.dynamic_buttons = [
            self.ui.cnnEegBtn,
            self.ui.ganMriBtn,
            self.ui.dbAdminBtn,
            self.ui.switchSceneBtn,
            self.ui.startBtn,
            self.ui.saveDbBtn
        ]

        self.ui.plotMetricsLbl.setPixmap(QPixmap(MAGNIFYING_GLASS_BRAIN_ICON_PATH))
        self.ui.generatedImgLbl.setPixmap(QPixmap(MAGNIFYING_GLASS_BRAIN_ICON_PATH))

        self.ui.startBtn.clicked.connect(self._train_runner)
        self.ui.exitBtn.clicked.connect(on_exit)
        self.ui.stopBtn.clicked.connect(self._stop_model)
        self.ui.saveDbBtn.clicked.connect(self._send_to_db_runner)

        self._update_info_dump()
        show_warning_alert("Generating synthetic data using GAN models can be a time-consuming process, "
                           "especially for large datasets. Please ensure your system has enough resources and "
                           "patience for the process to complete.")

        clear_temp_model_directory()

    def _update_info_dump(self):
        self.ui.infoDumpLbl.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files};'
            f' CONTROL: {self.loaded_control_files})\n'
            f'{self.current_channels} channels'
        )

    def _train_runner(self):
        import app.properties.mri_config as config
        self.ui.progressBar.setValue(0)
        self.ui.setFixedSize(self.ui.size())
        self.ui.statusLbl.setText("STATUS: Running")
        set_enabled_buttons_boolean(self.dynamic_buttons, False)

        epochs, batch_size, learning_rate, test_size, info_interval = get_main_train_ui_values(
            self.ui.epochsSpinBox,
            self.ui.batchSizeSpinBox,
            self.ui.learningRateSpinBox,
            self.ui.testSizeSpinBox,
            self.ui.infoIntervalSpinBox
        )
        config.GAN_EPOCHS_MRI = epochs
        config.GAN_BATCH_SIZE_MRI = batch_size
        config.GAN_LEARNING_RATE = learning_rate
        config.CNN_TEST_SIZE_MRI_GAN = test_size
        config.INFO_GAN_DISP_INTERVAL = info_interval
        create_admin_controller_thread(self)

    def _send_to_db_runner(self):
        from app.properties.mri_config import GAN_INPUT_SHAPE_MRI, GAN_LEARNING_RATE, GAN_BATCH_SIZE_MRI, GAN_EPOCHS_MRI
        input_shape = GAN_INPUT_SHAPE_MRI
        learning_rate = GAN_LEARNING_RATE
        num_of_electrodes = None
        batch_size = GAN_BATCH_SIZE_MRI
        epochs = GAN_EPOCHS_MRI
        model_type = 'gan_adhd' if self.ui.radioButtonAdhd.isChecked() else 'gan_control'
        fs = None
        plane = 'A'
        additional_info = (f"learning rate: {learning_rate}; batch size: {batch_size}; epochs: {epochs}; "
                           f"{self.ui.modelDescriptionTxtEdit.toPlainText()}")
        send_to_db(num_of_electrodes, self.db_conn, self.ui.dbStatusLbl, self.ui.statusLbl, input_shape, model_type, fs,
                   additional_info, plane)

    def _stop_model(self):
        if not self.training_manager.is_stopped():
            self.training_manager.stop_training()
            self.ui.statusLbl.setText("STATUS: Stopping...")

    def train(self):
        if self.ui.radioButtonControl.isChecked():
            train_gan("control", self.training_manager, self)
        elif self.ui.radioButtonAdhd.isChecked():
            train_gan("adhd", self.training_manager, self)

    def on_finished_runner(self):
        on_finished(self.ui.moreInfoLbl, self.ui.statusLbl, self.dynamic_buttons)

    def on_error(self, error):
        set_enabled_buttons_boolean(self.dynamic_buttons, True)
        print(f"Error: on_error - {error}")

    def update_gui_gan(self, epoch, metrics):
        update_gui_gan(self, epoch, metrics, self.ui.progressBar, self.ui.plotMetricsLbl, self.ui.generatedImgLbl)
