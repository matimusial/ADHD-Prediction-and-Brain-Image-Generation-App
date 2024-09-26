import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from app.training.management.training_callbacks import EarlyStoppingCallback, RealTimeMetricsCallback
from app.training.management.training_signal_functions import connect_signals
from app.services.data_processing.mri_processing import normalize_mri, trim_mri, prepare_for_cnn
from app.utils.file_utils import read_pickle


def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_cnn(manager_instance, controller_instance):
    """
    Trains a CNN model using ADHD and control MRI data, managing the training process and reporting metrics.

    Returns:
    - The test accuracy of the model after training.
    """
    from app.properties.mri_config import (
        CNN_EPOCHS_MRI, CNN_BATCH_SIZE_MRI, CNN_LEARNING_RATE_MRI, CNN_INPUT_SHAPE_MRI
    )
    from app.properties.directory_config import ADHD_MRI_REAL_PICKLE_PATH, CONTROL_MRI_REAL_PICKLE_PATH, TEMP_MODEL_PATH

    print(f"CNN training started for {CNN_EPOCHS_MRI} epochs...")
    print("\n")

    try:
        adhd_data = read_pickle(ADHD_MRI_REAL_PICKLE_PATH)
        control_data = read_pickle(CONTROL_MRI_REAL_PICKLE_PATH)
    except Exception as e:
        print(f"Error loading original files: {e}")
        return None

    # Preprocess ADHD and control data
    adhd_trimmed = trim_mri(adhd_data, CNN_INPUT_SHAPE_MRI[:2])
    adhd_normalized = normalize_mri(adhd_trimmed)

    control_trimmed = trim_mri(control_data, CNN_INPUT_SHAPE_MRI[:2])
    control_normalized = normalize_mri(control_trimmed)

    x_train, x_test, y_train, y_test = prepare_for_cnn(adhd_normalized, control_normalized)

    def fit_function():
        """
        Fits the CNN model by performing the training loop for the specified number of epochs.

        Returns:
        - The test accuracy of the model.
        """
        model = build_cnn_model(CNN_INPUT_SHAPE_MRI)
        model.compile(optimizer=Adam(learning_rate=CNN_LEARNING_RATE_MRI),
                      loss='binary_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)

        stop_training_callback = EarlyStoppingCallback(manager_instance)

        real_time_metrics = RealTimeMetricsCallback(total_epochs=CNN_EPOCHS_MRI)
        connect_signals(controller_instance, real_time_metrics)

        _ = model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      epochs=CNN_EPOCHS_MRI,
                      batch_size=CNN_BATCH_SIZE_MRI,
                      callbacks=[reduce_lr, real_time_metrics, stop_training_callback],
                      verbose=1)

        _, test_accuracy = model.evaluate(x_test, y_test)

        model.save(os.path.join(TEMP_MODEL_PATH, f'{round(test_accuracy, 4)}.keras'))
        return test_accuracy

    accuracy = fit_function()

    print(f"Test accuracy: {round(accuracy, 4)}")
    return round(accuracy, 4)
