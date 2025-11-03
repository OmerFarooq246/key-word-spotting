import os

class Config():
    def __init__(self):
        self.model_name = None
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.RESULTS_DIR = os.path.join(self.PROJECT_ROOT, "Results")

    @property
    def BEST_WEIGHTS_PATH(self):
        return f"{self.RESULTS_DIR}/{self.model_name}/best_weights.h5"
    
    @property
    def TRAIN_PLOT_PATH(self):
        return f"{self.RESULTS_DIR}/{self.model_name}/training_plot.png"
    
    @property
    def CLS_REPORT_PATH(self):
        return f"{self.RESULTS_DIR}/{self.model_name}/cls_report.txt"
    
    @property
    def CM_PATH(self):
        return f"{self.RESULTS_DIR}/{self.model_name}/cm.png"

    @property
    def MODEL_SUMMARY_PATH(self):
        return f"{self.RESULTS_DIR}/{self.model_name}/summary.txt"
        
    FRAME_LENGTH = 480      # 30 ms @ 16kHz, to set fft_length to 512 and not 1028
    FRAME_STEP = 240        # 15 ms stride
    FFT_LENGTH = 512		# power of 2, efficient computation
    NUM_MEL_BINS = 40
    NUM_MFCCS = 13			# standard for speech recognition
    LOWER_EDGE_HERTZ = 20
    UPPER_EDGE_HERTZ = 4000
    SR = 16000

    SEED = 42
    INPUT_SHAPE = (65, 13)
    NUM_CLASSES = 12
    BATCH_SIZE = 64

config = Config()