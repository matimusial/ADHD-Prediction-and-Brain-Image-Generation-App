
EEG_SIGNAL_FRAME_SIZE = 128
EEG_NUM_OF_ELECTRODES = 19
FS = 128

CUTOFFS = [(4,8), (12,30), (4,30)]  # Frequency [theta, beta, both]

CNN_INPUT_SHAPE = (EEG_NUM_OF_ELECTRODES, EEG_SIGNAL_FRAME_SIZE, 1)
CNN_EPOCHS = 5
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
TEST_SIZE_EEG_CNN = 0.2

ELECTRODE_POSITIONS = [
    "Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3", "F4",
    "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2"
]