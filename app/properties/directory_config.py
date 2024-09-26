import os

APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADMIN_DB_UI_PATH = os.path.join(APP_PATH, 'ui', 'admin', 'admin_db.ui')
ADMIN_EEG_CNN_UI_PATH = os.path.join(APP_PATH, 'ui', 'admin', 'admin_eeg_cnn.ui')
ADMIN_MRI_CNN_UI_PATH = os.path.join(APP_PATH, 'ui', 'admin', 'admin_mri_cnn.ui')
ADMIN_MRI_GAN_UI_PATH = os.path.join(APP_PATH, 'ui', 'admin', 'admin_mri_gan.ui')

DOCTOR_UI_PATH = os.path.join(APP_PATH, 'ui', 'doctor', 'doctor.ui')
DOCTOR_GEN_NEW_PIC_UI_PATH = os.path.join(APP_PATH, 'ui', 'doctor', 'doctor_gen_new_pic.ui')

MAGNIFYING_GLASS_CHART_ICON_PATH = os.path.join(APP_PATH, 'ui', 'icons', 'magnifying_glass_chart.png')
MAGNIFYING_GLASS_BRAIN_ICON_PATH = os.path.join(APP_PATH, 'ui', 'icons', 'magnifying_glass_brain.png')
LOADING_GIF_PATH = os.path.join(APP_PATH, 'ui', 'icons', 'loading.gif')

ADHD_MRI_REAL_PICKLE_PATH = os.path.join(APP_PATH, 'project_data', 'mri_real_files', 'adhd_mri_real.pkl')
CONTROL_MRI_REAL_PICKLE_PATH = os.path.join(APP_PATH, 'project_data', 'mri_real_files', 'control_mri_real.pkl')

ADHD_EEG_TRAIN_PICKLE_PATH = os.path.join(APP_PATH, 'project_data', 'eeg_train_data', 'adhd_eeg_train.pkl')
CONTROL_EEG_TRAIN_PICKLE_PATH = os.path.join(APP_PATH, 'project_data', 'eeg_train_data', 'control_eeg_train.pkl')

TEMP_MODEL_PATH = os.path.join(APP_PATH, 'project_data', 'temp_model')

REAL_MRI_DATA_PATH = os.path.join(APP_PATH, 'project_data', 'mri_real_files')
def FILE_PATH_FOR_PREPARE_AND_PLOT_DATA(data_type, radio_adhd):
    """
    :param data_type: real or generated
    :returns :app/project_data/mri_real_files/adhd_mri_real.pkl or
    app/project_data/mri_generated_files/control_mri_generated.pkl
    """
    file_path = os.path.join(APP_PATH,
                             'project_data', f'mri_{data_type}_files',
                             f"{'adhd' if radio_adhd.isChecked() else 'control'}_mri_{data_type}.pkl"
                             )
    return file_path

