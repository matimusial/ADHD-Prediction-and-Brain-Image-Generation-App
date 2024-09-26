import numpy as np


def check_result(predictions, threshold=0.5):
    mean_prediction = np.mean(predictions)
    if mean_prediction > threshold:
        result = "adhd"
        probability = np.round(mean_prediction * 100, 2)
    else:
        result = "control"
        probability = np.round((1 - mean_prediction) * 100, 2)

    print(f"Patient result: {result}, with probability: {probability}%")
    return result, probability
