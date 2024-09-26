from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QHeaderView
from PyQt5.QtCore import Qt

from app.services.database import Database
from app.services.gui_services.general import on_exit
from app.services.gui_services.alerts import show_warning_alert


class AdminDbController:

    def __init__(self, ui):
        self.ui = ui
        self.db = Database()
        self.id_table = []
        self.ui.deleteModelBtn.clicked.connect(self._show_dialog)
        self.ui.exitBtn.clicked.connect(on_exit)
        self._set_range_and_update_label()

    def _set_range_and_update_label(self):
        try:
            self.db.establish_connection()
        except ConnectionError as e:
            show_warning_alert(str(e))
            return

        data, column_names = self.db.select_data_and_columns("gan_models")

        if data:
            self.id_table = [record[0] for record in data]
            min_id, max_id = self.id_table[0], self.id_table[-1]
            self.ui.modelIdSpinBox.setRange(min_id, max_id)
            self._populate_table(data, column_names)
        else:
            show_warning_alert("No data available")

    def _populate_table(self, data, column_names):
        self.ui.modelTableWidget.setRowCount(len(data))
        self.ui.modelTableWidget.setColumnCount(len(column_names))
        self.ui.modelTableWidget.setHorizontalHeaderLabels(column_names)
        self.ui.modelTableWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled)
                self.ui.modelTableWidget.setItem(row_num, col_num, item)

        self.ui.modelTableWidget.resizeColumnsToContents()
        self.ui.modelTableWidget.resizeRowsToContents()

        header = self.ui.modelTableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setSectionResizeMode(column_names.index('description'), QHeaderView.Stretch)

        vertical_header = self.ui.modelTableWidget.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.Stretch)

        self.ui.modelTableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.ui.modelTableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def _delete_row(self, model_id):
        try:
            self.db.establish_connection()
        except ConnectionError as e:
            show_warning_alert(str(e))
            return

        self.db.delete_data_from_models_table(model_id)
        self._set_range_and_update_label()

    def _show_dialog(self):
        model_id = self.ui.modelIdSpinBox.value()

        if model_id not in self.id_table:
            show_warning_alert(f"ID: {model_id} does not exist!")
        else:
            alert = QMessageBox()
            alert.setWindowTitle("Warning")
            alert.setText(f"Are you sure you want to delete model {model_id}?")
            alert.setIcon(QMessageBox.Warning)
            alert.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            response = alert.exec_()
            if response == QMessageBox.Yes:
                self._delete_row(model_id)
