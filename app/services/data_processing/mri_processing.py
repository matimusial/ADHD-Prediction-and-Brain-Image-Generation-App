import copy

import numpy as np
from sklearn.model_selection import train_test_split


def normalize_mri(data):
    """
    Normalizes list of images to the range [-1, 1].
    """
    normalized = np.empty_like(data)
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        if max_val == min_val:
            normalized[i] = np.zeros_like(data[i])
        else:
            normalized[i] = (data[i] - min_val) / (max_val - min_val)
            normalized[i] = 2 * normalized[i] - 1
    return normalized


def trim_mri(data, target_size):
    """
    Trims image(s) to the target size by removing equal rows and columns from each side.
    Handles both single image and list of images.

    Parameters:
        - data: np.array (single image) or list of np.array (multiple images)
        - target_size: tuple (target_height, target_width)

    Returns:
        - np.array or list of np.array with resized images if possible.
    """

    def trim_image(image, target_height, target_width):
        current_height, current_width = image.shape[:2]

        if current_height < target_height or current_width < target_width:
            print(
                f"Error: Image size {current_height}x{current_width} too small to resize to {target_height}x{target_width}")
            return image

        trim_rows = (current_height - target_height) // 2
        trim_cols = (current_width - target_width) // 2

        trimmed_image = image[trim_rows:current_height - trim_rows, trim_cols:current_width - trim_cols]
        return trimmed_image

    target_height, target_width = target_size

    if isinstance(data, np.ndarray):
        return trim_image(data, target_height, target_width)

    elif isinstance(data, list):
        trimmed_images = []
        for i, img in enumerate(data):
            trimmed_images.append(trim_image(img, target_height, target_width))
        return trimmed_images

    else:
        print("Error: trim_mri - Unsupported data type")
        return None


def prepare_x_y(adhd_data, control_data, cnn_shape):
    """Prepares X and y data for a CNN model.

    Args:
        adhd_data (list or np.array): ADHD data.
        control_data (list or np.array): Control data.
        cnn_shape (int): The shape for CNN input.

    Returns:
        X (np.array): Combined ADHD and control data reshaped for CNN.
        y (np.array): Labels for ADHD and control data.
    """
    y_adhd = np.ones(len(adhd_data))
    y_control = np.zeros(len(control_data))
    y = np.hstack((y_adhd, y_control))

    x_adhd = np.reshape(adhd_data, (len(adhd_data), cnn_shape, cnn_shape, 1))
    x_control = np.reshape(control_data, (len(control_data), cnn_shape, cnn_shape, 1))
    x = np.vstack((x_adhd, x_control))

    return x, y


def prepare_for_cnn(adhd_data, control_data):
    """Prepares data for CNN.

    Args:
        adhd_data (list or np.array): Processed ADHD data.
        control_data (list or np.array): Processed control data.

    Returns:
        tuple: Split data ready for CNN (X_train, X_test, y_train, y_test).
    """
    from app.properties.mri_config import CNN_SINGLE_INPUT_SHAPE_MRI
    from app.properties.mri_config import TEST_SIZE_MRI_CNN

    x, y = prepare_x_y(adhd_data, control_data, CNN_SINGLE_INPUT_SHAPE_MRI)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE_MRI_CNN, shuffle=True)

    return x_train, x_test, y_train, y_test
