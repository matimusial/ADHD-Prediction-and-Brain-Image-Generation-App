import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from app.training.management.training_callbacks import RealTimeMetricsCallback, EarlyStoppingCallback
from app.properties.directory_config import TEMP_MODEL_PATH
from app.services.data_processing.eeg_processing import read_eeg_raw, filter_bandpass_eeg_data, normalize_eeg_data, \
    clip_eeg_extremes, prepare_for_cnn
from app.training.management.training_signal_functions import connect_signals


def build_eeg_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (10, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 1)),
        Conv2D(32, (8, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 1)),
        Conv2D(64, (4, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 1)),
        Flatten(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


def train_cnn_eeg(raw_data_path, manager_instance, controller_instance):
    """
    Trains a CNN model using EEG data, managing the training process and reporting metrics.

    Returns:
    - The test accuracy of the model after training.
    """
    from app.properties.cnn_config import CNN_INPUT_SHAPE, CNN_EPOCHS, CNN_LEARNING_RATE, CNN_BATCH_SIZE

    print(f"CNN training started for {CNN_EPOCHS} epochs...")
    print("\n")

    try:
        adhd_data, control_data, _, _, _ = read_eeg_raw(raw_data_path)
    except Exception as e:
        print(f"Error loading original files: {e}")
        return None

    # Preprocess EEG data
    adhd_filtered, control_filtered = filter_bandpass_eeg_data(adhd_data, control_data)
    adhd_clipped, control_clipped = clip_eeg_extremes(adhd_filtered, control_filtered)
    adhd_normalized, control_normalized = normalize_eeg_data(adhd_clipped, control_clipped)

    x_train, y_train, x_test, y_test = prepare_for_cnn(adhd_normalized, control_normalized)

    def fit_function():
        """
        Fits the CNN model by performing the training loop for the specified number of epochs.

        Returns:
        - The test accuracy of the model.
        """
        model = build_eeg_cnn_model(CNN_INPUT_SHAPE)
        model.compile(optimizer=Adam(learning_rate=CNN_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
        stop_training_callback = EarlyStoppingCallback(manager_instance)
        real_time_metrics = RealTimeMetricsCallback(total_epochs=CNN_EPOCHS)

        connect_signals(controller_instance, real_time_metrics)

        _ = model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      epochs=CNN_EPOCHS,
                      batch_size=CNN_BATCH_SIZE,
                      callbacks=[reduce_lr, real_time_metrics, stop_training_callback, early_stopping],
                      verbose=1
                      )

        _, test_accuracy = model.evaluate(x_test, y_test)

        model.save(os.path.join(TEMP_MODEL_PATH, f'{round(test_accuracy, 4)}.keras'))
        return test_accuracy

    accuracy = fit_function()

    print(f"Test accuracy: {round(accuracy, 4)}")
    return round(accuracy, 4)

