import os
import pickle
import shutil


def read_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error: read_pickle - {e}")


def clear_temp_model_directory():
    """
    Clears the temporary model directory by deleting all files and subdirectories within it.
    If the directory does not exist, it will be created.
    """
    from app.properties.directory_config import TEMP_MODEL_PATH

    if not os.path.exists(TEMP_MODEL_PATH):
        os.makedirs(TEMP_MODEL_PATH)

    try:
        for filename in os.listdir(TEMP_MODEL_PATH):
            file_path = os.path.join(TEMP_MODEL_PATH, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    except Exception as e:
        print(f"Error: delete_model - {e}")
