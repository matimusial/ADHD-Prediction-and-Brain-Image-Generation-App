from PyQt5.QtWidgets import QMessageBox


def show_warning_alert(msg):
    alert = QMessageBox()
    alert.setWindowTitle("Warning")
    alert.setText(msg)
    alert.setIcon(QMessageBox.Warning)
    alert.setStandardButtons(QMessageBox.Ok)
    alert.exec_()


def show_critical_alert(msg):
    alert = QMessageBox()
    alert.setIcon(QMessageBox.Critical)
    alert.setText(msg)
    alert.setWindowTitle("Critical")
    alert.setStandardButtons(QMessageBox.Ok)
    alert.exec_()


def show_info_alert(msg):
    alert = QMessageBox()
    alert.setIcon(QMessageBox.Information)
    alert.setText(msg)
    alert.setWindowTitle("Information")
    alert.setStandardButtons(QMessageBox.Ok)
    alert.exec_()


def handle_error(error_msg):
    show_warning_alert(f"Error: {error_msg}")
