import os

from PyQt5.QtCore import QSize

from app.services.gui_services.alerts import show_critical_alert
from app.utils.gui_utils import set_enabled_buttons_boolean


def get_main_train_ui_values(epochs_spinbox, batch_size_spinbox, learning_rate_spinbox, test_size_spinbox,
                             info_interval_spinbox=None):
    epochs = epochs_spinbox.value()
    batch_size = batch_size_spinbox.value()
    learning_rate = learning_rate_spinbox.value()
    test_size = test_size_spinbox.value()

    if not all([epochs, batch_size, learning_rate, test_size]):
        show_critical_alert("Invalid input")
        return None, None, None, None

    if info_interval_spinbox is not None:
        info_interval = info_interval_spinbox.value()
        return epochs, batch_size, learning_rate, test_size, info_interval

    return epochs, batch_size, learning_rate, test_size


def on_finished(more_info_label, status_label, dynamic_buttons):
    """
    Handles the GUI updates when model training is finished. Displays the final model accuracy or a warning
    if the model file is not found. Updates the status label and enables buttons.
    """
    from app.properties.directory_config import TEMP_MODEL_PATH

    file_name = os.listdir(TEMP_MODEL_PATH)
    if file_name:
        acc = file_name[0].replace(".keras", "")
        more_info_label.setText(f"Final model accuracy: {acc}")
    else:
        more_info_label.setText(f"Warning: Could not find model file in model_path")

    status_label.setText("STATUS: Model done")
    more_info_label.setFixedSize(QSize(16777215, 16777215))

    set_enabled_buttons_boolean(dynamic_buttons, True)

