import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QFontMetrics, QPixmap

from app.training.machine_learning.eeg.train_cnn_eeg import train_cnn_eeg
from app.services.database import Database
from app.services.data_processing.eeg_processing import read_eeg_raw
from app.training.management.training_signal_functions import update_gui
from app.services.gui_services.general import on_exit
from app.services.gui_services.training_gui import get_main_train_ui_values, on_finished
from app.services.gui_services.alerts import show_warning_alert
from app.utils.gui_utils import set_enabled_buttons_boolean
from app.utils.file_utils import clear_temp_model_directory
from app.services.database_services import send_to_db
from app.training.management.create_thread_functions import create_admin_controller_thread
from app.training.management.training_callbacks import TrainingManager


class AdminEegCnnController:
    def __init__(self, ui):

        from app.properties.directory_config import MAGNIFYING_GLASS_CHART_ICON_PATH
        from app.properties.cnn_config import FS

        self.ui = ui

        self.ui.cnnEegBtn.setEnabled(False)
        self.loaded_adhd_files = 0
        self.loaded_control_files = 0
        self.current_channels = 0
        self.current_samples = 0

        self.train_path = None
        self.worker = None
        self.thread = None
        self.real_time_metrics = None

        self.db_conn = Database()
        self.training_manager = TrainingManager()

        self.dynamic_buttons = [
            self.ui.cnnMriBtn,
            self.ui.ganMriBtn,
            self.ui.dbAdminBtn,
            self.ui.switchSceneBtn,
            self.ui.folderExploreBtn,
            self.ui.startBtn,
            self.ui.saveDbBtn
        ]

        self.ui.textEditElectrodesTxtEdit.setPlainText(str(self.current_channels))
        self.ui.textEditFrequencyTxtEdit.setPlainText(str(FS))

        self.ui.plotMetricsLbl.setPixmap(QPixmap(MAGNIFYING_GLASS_CHART_ICON_PATH))

        self.ui.folderExploreBtn.clicked.connect(self._show_dialog)
        self.ui.startBtn.clicked.connect(self._train_runner)
        self.ui.stopBtn.clicked.connect(self._stop_model)
        self.ui.exitBtn.clicked.connect(on_exit)
        self.ui.saveDbBtn.clicked.connect(self._send_to_db_runner)

        self._update_info_dump()
        clear_temp_model_directory()

    def _update_info_dump(self):
        self.ui.infoDumpLbl.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files};'
            f' CONTROL: {self.loaded_control_files})\n'
            f'{self.current_channels} channels; {self.current_samples} samples'
        )
        self.ui.textEditElectrodesTxtEdit.setPlainText(str(self.current_channels))

    def _train_runner(self):
        import app.properties.cnn_config as config
        self.ui.progressBar.setValue(0)
        self.ui.setFixedSize(self.ui.size())
        self.ui.statusLbl.setText("STATUS: Running")
        set_enabled_buttons_boolean(self.dynamic_buttons, False)

        epochs, batch_size, learning_rate, test_size = get_main_train_ui_values(
            self.ui.epochsSpinBox,
            self.ui.batchSizeSpinBox,
            self.ui.learningRateSpinBox,
            self.ui.testSizeSpinBox
        )
        config.CNN_EPOCHS = epochs
        config.CNN_BATCH_SIZE = batch_size
        config.CNN_LEARNING_RATE = learning_rate
        config.CNN_TEST_SIZE_EEG_CNN = test_size
        create_admin_controller_thread(self)

    def _send_to_db_runner(self):
        from app.properties.cnn_config import (EEG_NUM_OF_ELECTRODES, CNN_INPUT_SHAPE, FS, CNN_LEARNING_RATE,
                                               CNN_BATCH_SIZE, CNN_EPOCHS)
        input_shape = CNN_INPUT_SHAPE
        learning_rate = CNN_LEARNING_RATE
        num_of_electrodes = EEG_NUM_OF_ELECTRODES
        batch_size = CNN_BATCH_SIZE
        epochs = CNN_EPOCHS
        model_type = 'cnn_eeg'
        fs = FS
        plane = None
        additional_info = (f"learning rate: {learning_rate}; batch size: {batch_size}; epochs: {epochs}; "
                           f"{self.ui.modelDescriptionTxtEdit.toPlainText()}")
        send_to_db(num_of_electrodes, self.db_conn, self.ui.dbStatusLbl, self.ui.statusLbl, input_shape, model_type, fs,
                   additional_info, plane)

    def _stop_model(self):
        if not self.training_manager.is_stopped():
            self.training_manager.stop_training()
            self.ui.statusLbl.setText("STATUS: Stopping...")

    def train(self):
        train_cnn_eeg(self.train_path, self.training_manager, self)

    def on_finished_runner(self):
        on_finished(self.ui.moreInfoLbl, self.ui.statusLbl, self.dynamic_buttons)

    def on_error(self, error):
        set_enabled_buttons_boolean(self.dynamic_buttons, True)
        print(f"Error: on_error - {error}")

    def update_gui(self, epoch, metrics):
        update_gui(self, epoch, metrics, self.ui.progressBar, self.ui.plotMetricsLbl)

    def _show_dialog(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        if folder:
            adhd_path = os.path.join(folder, 'adhd')
            control_path = os.path.join(folder, 'control')

            if os.path.isdir(adhd_path) and os.path.isdir(control_path):
                self.train_path = folder
                metrics = QFontMetrics(self.ui.pathLbl.font())
                elided_text = metrics.elidedText(folder, Qt.ElideMiddle, self.ui.pathLbl.width())
                self.ui.pathLbl.setText(elided_text)
                _, _, init_channels, adhd_count, control_count = read_eeg_raw(folder)
                self.loaded_adhd_files = adhd_count
                self.loaded_control_files = control_count
                self.current_channels = init_channels[0]['shape'][0]
                self.current_samples = init_channels[0]['shape'][1]
                self._update_info_dump()
            else:
                show_warning_alert("Invalid input folder")
