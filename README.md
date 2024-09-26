# ADHD Prediction and Brain Image Generation App

## Overview

This is a desktop application developed using Python, PyQt5, and PostgreSQL, designed to predict ADHD from EEG data using a Convolutional Neural Network (CNN) and generate brain images using a Generative Adversarial Network (GAN). The application processes EEG and MRI data, enabling the model to identify ADHD patterns and augment the MRI dataset with generated images.

The current version of the code has been **heavily refactored** compared to the previous branch, resulting in a more structured, modular, and efficient implementation.

For a more detailed overview of the project, please visit the **Info** section of the project website.

## Features

- **ADHD Prediction from EEG Data**: The model is trained to predict ADHD by analyzing EEG data from children, both with and without ADHD.
- **MRI Image Generation**: A GAN model is employed to generate brain MRI images, enhancing the dataset used for model training.
- **Desktop Application**: Built using PyQt5, providing an intuitive graphical interface for managing data and predictions.
- **Local PostgreSQL Integration**: The application uses a local PostgreSQL database to manage and store the data.

## Datasets

### EEG Data

The EEG dataset is a collection of brain waves from children, both with and without ADHD. This data is crucial for training our model to identify patterns associated with ADHD.

**Source**: [EEG Data for ADHD and Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

### MRI Data

The MRI dataset provides brain images, which are used along with generated images to enhance the model's training and prediction capabilities.

**Source**: [OpenNeuro Dataset ds002424](https://openneuro.org/datasets/ds002424/versions/1.2.0)

## Functionality

- **EEG Data Processing and Model Training**: EEG data is processed into a format suitable for CNN analysis. Then, a deep learning model is trained to identify ADHD features from EEG patterns.
- **MRI Image Generation and Processing**: GANs are used to generate additional MRI images, augmenting the training dataset and improving model performance.

## Installation

1. Clone this repository.
2. Download the dataset files listed in the text files found in the corresponding folders.
3. Set up a local PostgreSQL database and update the configuration file (`database.py`) with your credentials.
4. Install the required Python packages:

   ```bash
   pip install PyQt5 numpy nibabel pyedflib scipy pandas matplotlib mne scikit-learn psycopg2 tensorflow
