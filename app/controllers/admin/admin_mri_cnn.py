from PyQt5.QtGui import QPixmap

from app.training.machine_learning.mri.train_cnn_mri import train_cnn
from app.services.database import Database
from app.training.management.training_signal_functions import update_gui
from app.services.gui_services.general import on_exit
from app.services.gui_services.training_gui import get_main_train_ui_values, on_finished
from app.utils.gui_utils import set_enabled_buttons_boolean
from app.utils.file_utils import clear_temp_model_directory
from app.services.database_services import send_to_db
from app.training.management.create_thread_functions import create_admin_controller_thread
from app.training.management.training_callbacks import TrainingManager
from app.utils.file_utils import read_pickle


class AdminMriCnnController:
    def __init__(self, ui):
        from app.properties.directory_config import (
            ADHD_MRI_REAL_PICKLE_PATH, CONTROL_MRI_REAL_PICKLE_PATH, MAGNIFYING_GLASS_CHART_ICON_PATH)

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

        self.ui.plotMetricsLbl.setPixmap(QPixmap(MAGNIFYING_GLASS_CHART_ICON_PATH))

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
            f'{self.current_channels} channels'
        )

    def _train_runner(self):
        import app.properties.mri_config as config
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
        config.CNN_EPOCHS_MRI = epochs
        config.CNN_BATCH_SIZE_MRI = batch_size
        config.CNN_LEARNING_RATE_MRI = learning_rate
        config.TEST_SIZE_MRI_CNN = test_size

        clear_temp_model_directory()

        create_admin_controller_thread(self)

    def _send_to_db_runner(self):
        from app.properties.mri_config import (CNN_INPUT_SHAPE_MRI, CNN_LEARNING_RATE_MRI, CNN_BATCH_SIZE_MRI,
                                               CNN_EPOCHS_MRI)
        input_shape = CNN_INPUT_SHAPE_MRI
        learning_rate = CNN_LEARNING_RATE_MRI
        num_of_electrodes = None
        batch_size = CNN_BATCH_SIZE_MRI
        epochs = CNN_EPOCHS_MRI
        model_type = 'cnn_mri'
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
        train_cnn(self.training_manager, self)

    def on_finished_runner(self):
        on_finished(self.ui.moreInfoLbl, self.ui.statusLbl, self.dynamic_buttons)

    def on_error(self, error):
        set_enabled_buttons_boolean(self.dynamic_buttons, True)
        print(f"Error: on_error - {error}")

    def update_gui(self, epoch, metrics):
        update_gui(self, epoch, metrics, self.ui.progressBar, self.ui.plotMetricsLbl)
