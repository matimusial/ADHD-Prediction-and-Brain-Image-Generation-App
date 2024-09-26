import random
import numpy as np
import nibabel as nib
import pyedflib

from PyQt5.QtCore import QModelIndex, QSize, Qt
from PyQt5.QtWidgets import (
    QFileDialog, QDialog, QVBoxLayout, QRadioButton, QLineEdit, QLabel,
    QPushButton
)
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QIntValidator, QMovie, QPixmap
from scipy.io import loadmat
from scipy.ndimage import rotate
from pandas import read_csv

from app.services.prediction_services.eeg_prediction import process_and_predict_eeg
from app.services.prediction_services.mri_prediction import process_and_predict_mri
from app.training.management.create_thread_functions import create_doctor_controller_thread
from app.utils.analysis_utils import check_result
from app.utils.file_utils import read_pickle
from app.services.database import Database
from app.utils.plotting.eeg_plot_utils import show_plot_eeg
from app.services.gui_services.general import on_exit
from app.services.gui_services.alerts import show_warning_alert, show_info_alert
from app.utils.plotting.mri_plot_utils import show_plot_mri
from app.utils.gui_utils import set_enabled_buttons_boolean


class DoctorController:

    def __init__(self, main_window, ui):
        from app.properties.directory_config import (
            MAGNIFYING_GLASS_BRAIN_ICON_PATH, MAGNIFYING_GLASS_CHART_ICON_PATH
        )

        self.main_window = main_window
        self.ui = ui

        self.db_conn = Database()
        self.model_eeg = None
        self.model_mri = None
        self.chosen_model_info_eeg = None
        self.chosen_model_info_mri = None
        self.predictions = None

        self.thread = None
        self.worker = None

        self.file_paths = []
        self.loaded_eeg_files = 0
        self.loaded_mri_files = 0
        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.curr_idx_channel = 0
        self.curr_idx_plane = 0

        self.all_data = {
            "eeg": {"data": [], "names": []},
            "mri": {"data": [], "names": []}
        }

        self.dynamic_buttons = [
            self.ui.predictBtn,
            self.ui.showGeneratedBtn,
            self.ui.switchSceneBtn,
            self.ui.loadDataBtn,
            self.ui.generateNewBtn,
            self.ui.showRealBtn
        ]

        self.ui.plotLblEEG.setPixmap(QPixmap(MAGNIFYING_GLASS_CHART_ICON_PATH))
        self.ui.plotLblMRI.setPixmap(QPixmap(MAGNIFYING_GLASS_BRAIN_ICON_PATH))
        self._add_events()

    def _add_events(self):
        self.ui.loadDataBtn.clicked.connect(self._get_file_paths)
        self.ui.nextPlotBtnEEG.clicked.connect(lambda: self._show_next_plot("eeg"))
        self.ui.prevPlotBtnEEG.clicked.connect(lambda: self._show_prev_plot("eeg"))
        self.ui.nextChannelBtnEEG.clicked.connect(lambda: self._show_next_channel_or_plane("eeg"))
        self.ui.prevChannelBtnEEG.clicked.connect(lambda: self._show_prev_channel_or_plane("eeg"))
        self.ui.nextPlotBtnMRI.clicked.connect(lambda: self._show_next_plot("mri"))
        self.ui.prevPlotBtnMRI.clicked.connect(lambda: self._show_prev_plot("mri"))
        self.ui.nextPlaneBtnMRI.clicked.connect(lambda: self._show_next_channel_or_plane("mri"))
        self.ui.prevPlaneBtnMRI.clicked.connect(lambda: self._show_prev_channel_or_plane("mri"))
        self.ui.modelInfoBtnEEG.clicked.connect(lambda: self._show_model_info("eeg"))
        self.ui.modelInfoBtnMRI.clicked.connect(lambda: self._show_model_info("mri"))
        self.ui.predictBtn.clicked.connect(self._predict_files)
        self.ui.showRealBtn.clicked.connect(lambda: self._show_dialog('real'))
        self.ui.showGeneratedBtn.clicked.connect(lambda: self._show_dialog('generated'))
        self.ui.exitBtn.clicked.connect(on_exit)

    def _get_file_paths(self):
        file_filter = "All supported files (*.mat *.csv *.edf *.nii.gz *.nii)"
        options = QFileDialog.Options()
        self.file_paths, _ = QFileDialog.getOpenFileNames(
            self.main_window, "Choose files", "", file_filter, options=options
        )

        if not self.file_paths:
            self.file_paths = []
            return

        self.loaded_eeg_files = sum(1 for path in self.file_paths if path.endswith(('.mat', '.edf', '.csv')))
        self.loaded_mri_files = sum(1 for path in self.file_paths if path.endswith(('.nii', '.nii.gz')))

        if self.loaded_eeg_files > 0:
            self.ui.imgViewer.setCurrentIndex(0)
        if self.loaded_mri_files > 0:
            self.ui.imgViewer.setCurrentIndex(1)

        self.ui.dataNameLbl.setText(f"{self.loaded_eeg_files} EEG and {self.loaded_mri_files} MRI files chosen")
        self._show_file_names()
        self._get_model_names()

    def _show_file_names(self):
        model_eeg = self.ui.fileListViewEEG.model()

        if model_eeg:
            model_eeg.clear()
        else:
            model_eeg = QStandardItemModel()

        if self.loaded_eeg_files > 0:
            for path in self.file_paths:
                if path.endswith(('.mat', '.edf', '.csv')):
                    item = QStandardItem(path.split("/")[-1])
                    item.setEditable(False)
                    item.setSelectable(False)
                    model_eeg.appendRow(item)

            self.ui.fileListViewEEG.setModel(model_eeg)

        model_mri = self.ui.fileListViewMRI.model()
        if model_mri:
            model_mri.clear()
        else:
            model_mri = QStandardItemModel()

        if self.loaded_mri_files > 0:
            for path in self.file_paths:
                if path.endswith(('.nii', '.nii.gz')):
                    item = QStandardItem(path.split("/")[-1])
                    item.setEditable(False)
                    item.setSelectable(False)
                    model_mri.appendRow(item)

            self.ui.fileListViewMRI.setModel(model_mri)

    def _get_model_names(self):
        self.chosen_model_info_eeg = None
        self.chosen_model_info_mri = None
        self.ui.chosenModelEEG.setText("----------")
        self.ui.chosenModelMRI.setText("----------")

        try:
            self.db_conn.establish_connection()
        except ConnectionError as e:
            show_warning_alert(str(e))
            return

        def _setup_model_view(model_list_view, model_type, chosen_model_info_attr, chosen_model_label_attr):
            model = model_list_view.model()
            if model:
                model.clear()
            else:
                model = QStandardItemModel()

            if getattr(self, f'loaded_{model_type}_files') > 0:
                def choose_model(index: QModelIndex):
                    item = model.itemFromIndex(index)
                    setattr(self, chosen_model_info_attr, item.data())
                    getattr(self.ui, chosen_model_label_attr).setText(item.data()[0])

                models_list = self.db_conn.select_model_info(f'cnn_{model_type}')

                for model_info in models_list:
                    item = QStandardItem(model_info[0])
                    item.setEditable(False)
                    item.setData(model_info)
                    model.appendRow(item)

                model_list_view.setModel(model)
                model_list_view.doubleClicked.connect(choose_model)

        _setup_model_view(self.ui.modelListViewEEG, 'eeg', 'chosen_model_info_eeg', 'chosenModelEEG')
        _setup_model_view(self.ui.modelListViewMRI, 'mri', 'chosen_model_info_mri', 'chosenModelMRI')

    def load_models(self):
        self.model_eeg = None
        self.model_mri = None

        if self.chosen_model_info_eeg is not None:
            self.model_eeg = self.db_conn.select_model(self.chosen_model_info_eeg[0])
        if self.chosen_model_info_mri is not None:
            self.model_mri = self.db_conn.select_model(self.chosen_model_info_mri[0])

    def _show_model_info(self, model_type):
        if model_type == "eeg":
            if self.chosen_model_info_eeg is None:
                return
            msg = f"""
                Model accuracy: {self.chosen_model_info_eeg[0]}
                Input shape: {self.chosen_model_info_eeg[1]}
                Frequency: {self.chosen_model_info_eeg[2]}
                Channels: {self.chosen_model_info_eeg[3]}
                Description: {self.chosen_model_info_eeg[5]}
            """
        elif model_type == "mri":
            if self.chosen_model_info_mri is None:
                return
            plane = self.chosen_model_info_mri[4]
            msg = f"""
                Model accuracy: {self.chosen_model_info_mri[0]}
                Input shape: {self.chosen_model_info_mri[1]}
                Plane: {'Axial' if plane == 'A' else 'Sagittal' if plane == 'S' else 'Coronal'}
                Description: {self.chosen_model_info_mri[5]}
            """
        show_info_alert(msg)

    def _predict_files(self):
        try:
            self.ui.nextPlaneBtnMRI.setEnabled(True)
            self.ui.prevPlaneBtnMRI.setEnabled(True)

            if not self.file_paths or (
                    self.chosen_model_info_eeg is None and self.loaded_eeg_files > 0) or (
                    self.chosen_model_info_mri is None and self.loaded_mri_files > 0):
                show_warning_alert("No files or models chosen")
                return

            self.curr_idx_eeg = 0
            self.curr_idx_mri = 0
            self.curr_idx_channel = 0
            self.curr_idx_plane = 0

            create_doctor_controller_thread(self)

            self._show_loading_animation()
            set_enabled_buttons_boolean(self.dynamic_buttons, False)
        except Exception as e:
            print(e)

    def process_files(self):
        """
        Processes a list of file paths containing EEG or MRI data. Depending on the file type, it loads and processes the data,
        applies the relevant model for prediction, and stores the results.

        Supported file types:
        - .edf: EEG data in EDF format.
        - .mat: EEG data in MATLAB format.
        - .csv: EEG data in CSV format.
        - .nii or .nii.gz: MRI data in NIfTI format.

        Predictions are stored in self.predictions and the processed data in self.all_data.
        """
        self.predictions = []
        self.all_data = {"eeg": {"data": [], "names": []}, "mri": {"data": [], "names": []}}

        for path in self.file_paths:
            data = np.array([])
            data_type = ""
            try:
                # Process EEG data in EDF format
                if path.endswith('.edf'):
                    f = pyedflib.EdfReader(path)
                    n_channels = f.signals_in_file
                    n_samples = f.getNSamples()[0]
                    data = np.zeros((n_channels, n_samples))
                    for i in range(n_channels):
                        data[i, :] = f.readSignal(i)
                    f.close()
                    data_type = "eeg"
                    model = self.model_eeg
                    model_info = self.chosen_model_info_eeg

                # Process EEG data in MATLAB format
                elif path.endswith('.mat'):
                    file = loadmat(path)
                    data_key = list(file.keys())[-1]
                    data = file[data_key].T
                    data_type = "eeg"
                    model = self.model_eeg
                    model_info = self.chosen_model_info_eeg

                # Process EEG data in CSV format
                elif path.endswith('.csv'):
                    data = read_csv(path).values.T
                    data_type = "eeg"
                    model = self.model_eeg
                    model_info = self.chosen_model_info_eeg

                # Process MRI data in NIfTI format
                elif path.endswith('.nii.gz') or path.endswith('.nii'):
                    file = nib.load(path)
                    file_data = file.get_fdata()
                    (x, y, z, t) = file_data.shape

                    # Extract different MRI planes
                    frontal_plane = file_data[:, int(y / 2), :, int(t / 2)]
                    sagittal_plane = file_data[int(x / 2), :, :, int(t / 2)]
                    horizontal_plane = file_data[:, :, int(z / 2), int(t / 2)]

                    data = horizontal_plane
                    data_type = "mri"
                    self.all_data[data_type]["data"].append([
                        horizontal_plane, rotate(sagittal_plane, 90, reshape=True),
                        rotate(frontal_plane, 90, reshape=True)
                    ])
                    model = self.model_mri
                    model_info = self.chosen_model_info_mri

                if data_type == "mri":
                    result = process_and_predict_mri(data, model)
                elif data_type == "eeg":
                    result = process_and_predict_eeg(data, model, model_info)

                if data_type == "eeg":
                    self.all_data[data_type]["data"].append(data)

                self.all_data[data_type]["names"].append(path)

                if result is not None:
                    self.predictions.append(result)
            except Exception as e:
                print(f"Error: process_files - {e}")

    def on_finished(self):
        set_enabled_buttons_boolean(self.dynamic_buttons, True)
        self.movie.stop()
        self._show_result()

    def on_error(self, error):
        set_enabled_buttons_boolean(self.dynamic_buttons, True)
        self.movie.stop()
        self._show_result()
        show_warning_alert(f"Error: {error}")

    def _show_loading_animation(self):
        from app.properties.directory_config import LOADING_GIF_PATH
        self.movie = QMovie(LOADING_GIF_PATH)
        self.ui.resultLbl.setMovie(self.movie)
        self.movie.setScaledSize(QSize(40, 40))
        self.movie.start()

    def _show_result(self):
        if not self.predictions:
            self.ui.resultLbl.setText("-----------")
            return

        predictions_means = [np.mean(prediction) for prediction in self.predictions]
        result, prob = check_result(np.array(predictions_means))

        self.ui.resultLbl.setText(f"{result} ({prob}%)")

        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.ui.plotLblEEG.clear()
        self.ui.plotLblMRI.clear()

        if self.all_data["eeg"]["data"]:
            self._show_plot(self.all_data["eeg"]["data"][0], "eeg",
                            self.all_data["eeg"]["names"][self.curr_idx_eeg].split("/")[-1])

        if self.all_data["mri"]["data"]:
            self._show_plot(self.all_data["mri"]["data"][0][0], "mri",
                            self.all_data["mri"]["names"][self.curr_idx_mri].split("/")[-1])

    def _show_dialog(self, data_type):
        dialog = QDialog()
        dialog.setWindowTitle('Choose option')
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        layout = QVBoxLayout()

        if data_type == 'generated':
            label_desc = QLabel('MRI pictures generated on default, built-in model.')
            layout.addWidget(label_desc)

        radio_adhd = QRadioButton('ADHD')
        radio_control = QRadioButton('CONTROL')
        radio_adhd.setChecked(True)

        layout.addWidget(radio_adhd)
        layout.addWidget(radio_control)

        label_img_amount = QLabel('IMG amount (max 20):')
        layout.addWidget(label_img_amount)

        input_number = QLineEdit()
        validator = QIntValidator(0, 20, input_number)
        input_number.setValidator(validator)
        input_number.setText("3")
        layout.addWidget(input_number)

        submit_button = QPushButton('Submit')
        submit_button.clicked.connect(
            lambda: self._prepare_and_plot_data(data_type, radio_adhd, input_number, dialog))
        layout.addWidget(submit_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def _prepare_and_plot_data(self, data_type, radio_adhd, input_number, dialog):
        """
        Prepares and plots MRI data based on the user input. Initializes indices for EEG and MRI data,
        loads data from a specified file, and selects a random subset for plotting.

        Parameters:
        - data_type: A string representing the type of data ("eeg" or "mri").
        - radio_adhd: A boolean or similar control indicating if ADHD data should be selected.
        - input_number: A control or input field specifying how many data samples to use.
        - dialog: The dialog window that is closed after selection.
        """
        from app.properties.directory_config import FILE_PATH_FOR_PREPARE_AND_PLOT_DATA

        self.ui.nextPlaneBtnMRI.setEnabled(False)
        self.ui.prevPlaneBtnMRI.setEnabled(False)

        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.curr_idx_channel = 0
        self.curr_idx_plane = 0

        self.all_data = {"eeg": {"data": [], "names": []}, "mri": {"data": [], "names": []}}

        dialog.close()

        file_path = FILE_PATH_FOR_PREPARE_AND_PLOT_DATA(data_type, radio_adhd)

        data = read_pickle(file_path)

        # Limit the number of images to the minimum of input_number and 20
        input_number = min(int(input_number.text()), 20)
        img_numbers = random.sample(range(len(data)), input_number)

        # Process and prepare MRI data
        for img_number in img_numbers:
            try:
                self.all_data["mri"]["data"].append([
                    data[img_number], np.zeros(data[img_number].shape), np.zeros(data[img_number].shape)
                ])
            except Exception as e:
                print(f"Error: prepare_and_plot_data - {e}")

        self._show_plot(self.all_data["mri"]["data"][0][0], "mri", "")

    def _show_plot(self, data, data_type, name=""):
        if data_type == "eeg":
            qpm = show_plot_eeg(data, name, self.curr_idx_channel)
            self.ui.plotLblEEG.setPixmap(qpm)
        if data_type == "mri":
            qpm = show_plot_mri(data, name)
            self.ui.plotLblMRI.setPixmap(qpm)

    def _update_index(self, idx_name, max_value, direction=1):
        if not self.all_data["eeg"]["data"] and 'eeg' in idx_name:
            return
        if not self.all_data["mri"]["data"] and 'mri' in idx_name:
            return

        setattr(self, idx_name, getattr(self, idx_name) + direction)
        setattr(self, idx_name, max(0, min(getattr(self, idx_name), max_value)))

    def _show_plot_generic(self, data_type, idx_data, idx_plane=None):
        """
        Displays a plot for either EEG or MRI data based on the provided data type and index.

        Parameters:
        - data_type: A string indicating the type of data ("eeg" or "mri").
        - idx_data: Index to select the specific data to display.
        - idx_plane: (Optional) Index to select the specific MRI plane, required if data_type is "mri".
        """
        if data_type == "eeg":
            data = self.all_data["eeg"]["data"][idx_data]
            name = self.all_data["eeg"]["names"][idx_data].split("/")[-1]
        elif data_type == "mri":
            data = self.all_data["mri"]["data"][idx_data][idx_plane]
            name = self.all_data["mri"]["names"][idx_data].split("/")[-1] if self.file_paths else ""

        self._show_plot(data, data_type, name)

    def _show_next_plot(self, data_type):
        if data_type == "eeg":
            self._update_index("curr_idx_eeg", len(self.all_data["eeg"]["data"]) - 1, direction=1)
            self._show_plot_generic("eeg", self.curr_idx_eeg)
        elif data_type == "mri":
            self._update_index("curr_idx_mri", len(self.all_data["mri"]["data"]) - 1, direction=1)
            self._show_plot_generic("mri", self.curr_idx_mri, self.curr_idx_plane)

    def _show_prev_plot(self, data_type):
        if data_type == "eeg":
            self._update_index("curr_idx_eeg", len(self.all_data["eeg"]["data"]) - 1, direction=-1)
            self._show_plot_generic("eeg", self.curr_idx_eeg)
        elif data_type == "mri":
            self._update_index("curr_idx_mri", len(self.all_data["mri"]["data"]) - 1, direction=-1)
            self._show_plot_generic("mri", self.curr_idx_mri, self.curr_idx_plane)

    def _show_next_channel_or_plane(self, data_type):
        if data_type == "eeg":
            self._update_index("curr_idx_channel", 18, direction=1)
            self._show_plot_generic("eeg", self.curr_idx_eeg)
        elif data_type == "mri":
            self._update_index("curr_idx_plane", 2, direction=1)
            self._show_plot_generic("mri", self.curr_idx_mri, self.curr_idx_plane)

    def _show_prev_channel_or_plane(self, data_type):
        if data_type == "eeg":
            self._update_index("curr_idx_channel", 18, direction=-1)
            self._show_plot_generic("eeg", self.curr_idx_eeg)
        elif data_type == "mri":
            self._update_index("curr_idx_plane", 2, direction=-1)
            self._show_plot_generic("mri", self.curr_idx_mri, self.curr_idx_plane)
