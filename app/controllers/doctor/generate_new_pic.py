import io

import numpy as np

from PyQt5.QtCore import QModelIndex, QSize
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPixmap, QMovie
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from app.services.database import Database
from app.training.management.create_thread_functions import create_generate_new_controller_thread
from app.services.gui_services.general import on_exit
from app.services.gui_services.alerts import show_warning_alert, show_info_alert
from app.utils.plotting.mri_plot_utils import show_plot_mri


class GenerateNewController:
    def __init__(self, ui):
        from app.properties.directory_config import MAGNIFYING_GLASS_BRAIN_ICON_PATH

        self.ui = ui
        self.db = Database()

        self.ax = None
        self.movie = None
        self.chosen_model_data = None
        self.generator = None
        self.thread = None
        self.worker = None
        self.fig = None

        self.generated = []
        self.curr_idx_mri = 0

        self.ui.imgNumberSpinBox.setRange(1, 20)
        self.ui.plotMriLbl.setPixmap(QPixmap(MAGNIFYING_GLASS_BRAIN_ICON_PATH))

        self._choose_model()
        self._add_events()

    def __del__(self):
        if self.generated:
            del self.generated
        if self.generator:
            del self.generator

    def _add_events(self):
        self.ui.adhdGenInfoBtn.clicked.connect(lambda: self._show_info("adhd"))
        self.ui.controlGenInfoBtn.clicked.connect(lambda: self._show_info("control"))
        self.ui.genBtn.clicked.connect(self._generate_img)
        self.ui.prevPlotMriBtn.clicked.connect(self._show_prev_plot_mri)
        self.ui.nextPlotMriBtn.clicked.connect(self._show_next_plot_mri)
        self.ui.savePlotBtn.clicked.connect(self._save_image)
        self.ui.exitBtn.clicked.connect(on_exit)

    def _choose_model(self):

        def _choose_adhd_model(index: QModelIndex):
            item = adhd_model.itemFromIndex(index)
            self.chosen_model_data = item.data()
            self.ui.adhdGenNameLbl.setText(self.chosen_model_data[0])

            self.ui.controlGenListView.clearSelection()
            self.ui.controlGenNameLbl.setText("---------------------------")

        def _choose_control_model(index: QModelIndex):
            item = control_model.itemFromIndex(index)
            self.chosen_model_data = item.data()
            self.ui.controlGenNameLbl.setText(self.chosen_model_data[0])

            self.ui.adhdGenListView.clearSelection()
            self.ui.adhdGenNameLbl.setText("---------------------------")

        try:
            self.db.establish_connection()
        except ConnectionError as e:
            show_warning_alert(str(e))
            return

        adhd_model = QStandardItemModel()
        adhd_list = self.db.select_model_info('gan_adhd')

        for item in adhd_list:
            item = list(item)
            item.append('adhd')
            adhd_item = QStandardItem(item[0])
            adhd_item.setEditable(False)
            adhd_item.setData(item)
            adhd_model.appendRow(adhd_item)
        self.ui.adhdGenListView.setModel(adhd_model)
        self.ui.adhdGenListView.doubleClicked.connect(_choose_adhd_model)

        control_model = QStandardItemModel()
        control_list = self.db.select_model_info('gan_control')

        for item in control_list:
            item = list(item)
            item.append('healthy')
            control_item = QStandardItem(item[0])
            control_item.setEditable(False)
            control_item.setData(item)
            control_model.appendRow(control_item)
        self.ui.controlGenListView.setModel(control_model)
        self.ui.controlGenListView.doubleClicked.connect(_choose_control_model)

    def _show_info(self, data):

        if data == "adhd" and self.ui.adhdGenNameLbl.text() == "---------------------------":
            return
        elif data == "control" and self.ui.controlGenNameLbl.text() == "---------------------------":
            return

        plane = self.chosen_model_data[4]
        msg = f"""
        Generator loss: {self.chosen_model_data[0]}
        Image size: {self.chosen_model_data[1]}
        Plane: {'Axial' if plane == 'A' else 'Sagittal' if plane == 'S' else 'Coronal'}
        Description: {self.chosen_model_data[5]}
        """

        show_info_alert(msg.strip())

    def _generate_img(self):

        self.generated = []
        self.curr_idx_mri = 0

        if self.chosen_model_data is None:
            show_warning_alert("Please select model!")
            return

        try:
            create_generate_new_controller_thread(self)
            self._show_loading_animation()
            self.ui.genBtn.setEnabled(False)
            self.ui.backBtn.setEnabled(False)
        except Exception as e:
            print(f"Error: generate - {e}")

    def on_model_loaded(self, model):

        try:
            self.generator = model

            img_amount = self.ui.imgNumberSpinBox.value()
            for _ in range(img_amount):
                noise = np.random.normal(0, 1, [1, 100])
                generated_image = self.generator.predict(noise)
                generated_image = generated_image * 0.5 + 0.5
                self.generated.append(generated_image[0])

            self.fig = show_plot_mri(self.generated[0])
            self.ui.plotMriLbl.setPixmap(self.fig)

            self.ui.genBtn.setEnabled(True)
            self.ui.backBtn.setEnabled(True)
        except Exception as e:
            print(f"Error: on_model_loaded - {e}")

    def _update_mri_index(self, direction=1):
        if not self.generated:
            return

        self.curr_idx_mri += direction
        self.curr_idx_mri = max(0, min(self.curr_idx_mri, len(self.generated) - 1))

        self.fig = show_plot_mri(self.generated[self.curr_idx_mri])
        self.ui.plotMriLbl.setPixmap(self.fig)

    def _show_prev_plot_mri(self):
        self._update_mri_index(direction=-1)

    def _show_next_plot_mri(self):
        self._update_mri_index(direction=1)

    def _show_loading_animation(self):
        from app.properties.directory_config import LOADING_GIF_PATH
        self.movie = QMovie(LOADING_GIF_PATH)
        self.ui.plotMriLbl.setMovie(self.movie)
        self.movie.setScaledSize(QSize(50, 50))
        self.movie.start()

    def _save_image(self):
        if not self.generated or self.curr_idx_mri >= len(self.generated):
            show_warning_alert("No MRI image available to save.")
            return

        current_image = self.generated[self.curr_idx_mri]
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.imshow(current_image, cmap="gray")
        ax.set_title(f"MRI Image {self.curr_idx_mri + 1}")

        canvas = FigureCanvas(fig)

        default_filename = f"MRI_Image_{self.curr_idx_mri + 1}.png"

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Figure", default_filename,
            "PNG Files (*.png);;All Files (*)", options=options
        )

        if file_path:
            buf = io.BytesIO()
            canvas.print_png(buf)
            with open(file_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.close()
