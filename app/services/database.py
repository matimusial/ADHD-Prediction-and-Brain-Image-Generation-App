import os
import tempfile

import psycopg2
from psycopg2 import sql


def _validate_and_convert_input_models(name, file_data, channels, input_shape, type_value, fs, plane,
                                       description):
    """
    Parameters:
        - name: Model name, must consist of digits and dots.
        - file_data: Path to the model file (.keras).
        - channels: Number of channels in the model (optional).
        - input_shape: The input shape of the model.
        - type_value: The type of model (cnn_mri, cnn_eeg, etc.).
        - fs: Sampling frequency (optional).
        - plane: MRI plane (A, S, C).
        - description: Description of the model.
    """
    if not all(char.isdigit() or char == '.' for char in name):
        print("Error: 'name' must consist only of digits (0-9) and/or dots ('.').")
        return

    if not file_data.endswith('.keras'):
        print("Error: 'model path' must have '.keras' extension.")
        return

    if channels is not None and not isinstance(channels, int):
        print("Error: 'channels' must be an integer or None.")
        return

    if not (isinstance(input_shape, tuple) and all(isinstance(i, int) for i in input_shape)):
        print("Error: 'input_shape' must be a tuple of integers.")
        return

    type_value = type_value.lower()
    if type_value not in ['cnn_mri', 'cnn_eeg', 'gan_adhd', 'gan_control']:
        print("Error: 'type' must be one of 'cnn_mri', 'cnn_eeg', 'gan_adhd', 'gan_control'.")
        return

    if fs is not None:
        if not isinstance(fs, (float, int)):
            print("Error: 'fs' must be a float, integer, or None.")
            return
        if fs == 0:
            print("Error: 'fs' cannot be 0.")
            return

    if plane is not None and plane not in ['A', 'S', 'C']:
        print("Error: 'plane' must be one of 'A', 'S', 'C' or None.")
        return

    if len(description) > 254:
        print("Error: 'description' cannot be longer than 254 characters.")
        return

    return name, file_data, channels, str(input_shape), type_value, float(fs) if fs is not None else None, plane, description


class Database:

    def __init__(self):
        self.connection = None
        self.cursor = None

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def establish_connection(self):
        if self.connection:
            try:
                self.cursor.execute('SELECT 1')
                return
            except psycopg2.Error:
                self.connection = None

        try:
            self.connection = psycopg2.connect(
                host="localhost",
                dbname="my_database",
                user="postgres",
                password="user"
            )
            self.cursor = self.connection.cursor()
        except psycopg2.Error as e:
            print(f"Error: establish_connection - {e}")

    def insert_data_into_models_table(self, name="none", file_path="none", channels=None, input_shape=(0, 0, 0),
                                      type_value="none", fs=None, plane=None, description="none"):

        validated_data = _validate_and_convert_input_models(name, file_path, channels, input_shape, type_value, fs,
                                                            plane, description)
        if not validated_data:
            print("Error: insert_data_into_models_table - Invalid input data.")

        name, file_path, channels, input_shape_str, type_value, fs, plane, description = validated_data

        try:
            with open(file_path, 'rb') as file:
                model_data = file.read()
        except Exception as e:
            print(f"Error: insert_data_into_models_table - {e}")

        try:
            query_models = """
            INSERT INTO gan_models (name, channels, input_shape, type, fs, plane, description) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            self.cursor.execute(query_models, (name, channels, input_shape_str, type_value, fs, plane, description))
            model_id = self.cursor.fetchone()[0]
            self.connection.commit()

            query_files = """
            INSERT INTO files (model_id, file) 
            VALUES (%s, %s)
            """
            self.cursor.execute(query_files, (model_id, model_data))
            self.connection.commit()

            print(f"Model {type_value} {name} successfully inserted into 'gan_models' and 'files' tables.")
        except psycopg2.Error as e:
            print(f"Error: insert_data_into_models_table - {e}")

    def select_data_and_columns(self, table_name="gan_models"):
        """
        Arguments:
            - table_name (str): The name of the table from which to select data (default is "gan_models").

        Returns:
            A tuple containing:
            - results: A list of tuples, where each tuple represents a row from the specified table.
            - column_names: A list of strings representing the column names of the table.
        """
        try:
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            column_names = [column[0] for column in self.cursor.description]
            return results, column_names
        except psycopg2.Error as e:
            print(f"Error: select_data_and_columns - {e}")

    def select_model_info(self, condition_value=""):
        """
         Arguments:
            - condition_value (str): The condition used to filter models by type (e.g., 'cnn_mri', 'gan_adhd').

        Returns:
            A list of tuples where each tuple contains the following fields:
            - name: The name of the model.
            - input_shape: The input shape of the model.
            - fs: The sampling frequency (if applicable).
            - channels: The number of channels (if applicable).
            - plane: The MRI plane (if applicable).
            - description: A description of the model.
        """
        try:
            query = (f"SELECT name, input_shape, fs, channels, plane, description FROM gan_models WHERE"
                     f" type = %s ORDER BY name DESC")
            self.cursor.execute(query, (condition_value,))
            results = self.cursor.fetchall()
            return results
        except psycopg2.Error as e:
            print(f"Error: select_model_info - {e}")

    def select_model(self, model_name=""):
        """
        Selects and returns the model from the database based on the model name.

        Returns:
            - Loaded Keras model.
        """
        from tensorflow.keras.models import load_model
        try:
            self.cursor.execute("SELECT id FROM gan_models WHERE name=%s", (model_name,))
            model_id_result = self.cursor.fetchone()

            if model_id_result:
                model_id = model_id_result[0]
                self.cursor.execute("SELECT file FROM files WHERE model_id=%s", (model_id,))
                file_result = self.cursor.fetchone()

                if file_result:
                    model_file_data = file_result[0]

                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_model_path = os.path.join(temp_dir, "tmp.keras")

                        with open(temp_model_path, 'wb') as file:
                            file.write(model_file_data)

                        try:
                            loaded_model = load_model(temp_model_path)
                            return loaded_model
                        except Exception as e:
                            print(f"Error: select_model - {e}")
                else:
                    print("Error: select_model - No file found for the specified model.")
            else:
                print("Error: select_model - No model found with the specified name.")
        except psycopg2.Error as e:
            print(f"Error: select_model - {e}")
        except IOError as e:
            print(f"Error: select_model - {e}")

    def delete_data_from_models_table(self, model_id):
        try:
            self.cursor.execute("DELETE FROM files WHERE model_id = %s", (model_id,))
            self.cursor.execute("DELETE FROM gan_models WHERE id = %s", (model_id,))
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"Error: delete_data_from_models_table - {e}")


"""
db = Database()
db.establish_connection()
db.insert_data_into_models_table(name="0.9307", file_path="../project_data/eeg_models/0.9307_eeg.keras", channels=19, input_shape=(19, 128, 1), type_value="cnn_eeg", fs=128, plane=None, description="learning rate: 0.001; batch size: 32; epochs: 20; default")
db.insert_data_into_models_table(name="0.8836", file_path="../project_data/mri_cnn_models/0.8836_mri.keras", channels=19, input_shape=(120, 128, 1), type_value="cnn_mri", fs=128, plane=None, description="learning rate: 0.001; batch size: 32; epochs: 8; default")
db.insert_data_into_models_table(name="2.3397", file_path="../project_data/gan_models/2.3397_adhd_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_adhd", fs=None, plane="A", description="learning rate: 0.0002; batch size: 150000; epochs: 150000; default")
db.insert_data_into_models_table(name="1.8406", file_path="../project_data/gan_models/1.8406_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_control", fs=None, plane="A", description="learning rate: 0.0002; batch size: 150000; epochs: 150000; default")
db.insert_data_into_models_table(name="1.9906", file_path="../project_data/gan_models/1.9474_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_adhd", fs=None, plane="A", description="learning rate: 0.0001; batch size: 128; epochs: 100000")
db.insert_data_into_models_table(name="2.6528", file_path="../project_data/gan_models/2.6528_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_control", fs=None, plane="A", description="learning rate: 0.0002; batch size: 150000; epochs: 100000")
db.insert_data_into_models_table(name="2.1986", file_path="../project_data/gan_models/2.1986_adhd_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_adhd", fs=None, plane="A", description="learning rate: 0.0002; batch size: 150000; epochs: 100000")
db.insert_data_into_models_table(name="1.9474", file_path="../project_data/gan_models/1.9474_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_control", fs=None, plane="A", description="learning rate: 0.0001; batch size: 32; epochs: 300000")
db.insert_data_into_models_table(name="3.1001", file_path="../project_data/gan_models/3.1001_adhd_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_adhd", fs=None, plane="A", description="learning rate: 0.00015; batch size: 128; epochs: 175000")
db.insert_data_into_models_table(name="3.8234", file_path="../project_data/gan_models/3.8234_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_control", fs=None, plane="A", description="learning rate: 0.00015; batch size: 128; epochs: 175000")
db.insert_data_into_models_table(name="4.3616", file_path="../project_data/gan_models/4.3616_adhd_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_adhd", fs=None, plane="A", description="learning rate: 0.00005; batch size: 64; epochs: 150000")
db.insert_data_into_models_table(name="5.1636", file_path="../project_data/gan_models/5.1636_control_gan.keras", channels=None, input_shape=(120, 120, 1), type_value="gan_control", fs=None, plane="A", description="learning rate: 0.00005; batch size: 64; epochs: 150000")
"""

"""
CREATE TABLE gan_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    channels INTEGER,
    input_shape VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    fs FLOAT,
    plane CHAR(1),
    description VARCHAR(255)
);

CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES gan_models(id) ON DELETE CASCADE,)
    file BYTEA NOT NULL
);
"""