import os

from app.services.gui_services.alerts import show_warning_alert, show_info_alert
from app.utils.file_utils import clear_temp_model_directory


def send_to_db(num_of_electrodes, db_conn, db_status_label, status_label, input_shape, model_type, fs, additional_info,
               plane):
    from app.properties.directory_config import TEMP_MODEL_PATH
    if os.path.exists(TEMP_MODEL_PATH) and os.listdir(TEMP_MODEL_PATH):
        file_name = os.listdir(TEMP_MODEL_PATH)[0]
        file_path = os.path.join(TEMP_MODEL_PATH, file_name)
        db_status_label.setText("STATUS: Connecting...")
        try:
            db_conn.establish_connection()
        except ConnectionError as e:
            show_warning_alert(str(e))
            return

        db_status_label.setText("STATUS: Sending...")
        status_label.setText("STATUS: Uploading model")
        try:
            db_conn.insert_data_into_models_table(
                file_name.replace(".keras", ""),
                file_path,
                num_of_electrodes,
                input_shape,
                model_type,
                fs,
                plane,
                additional_info
            )
            show_info_alert("Data successfully sent to database.")
        except Exception as e:
            show_warning_alert("Model upload has failed.")
            print(f"Error: send_to_db - Failed to upload model to db. Reason: {e}")

        status_label.setText("STATUS: Await")
        db_status_label.setText("STATUS: Await")
        clear_temp_model_directory()
    else:
        show_warning_alert("Model upload has failed.")
