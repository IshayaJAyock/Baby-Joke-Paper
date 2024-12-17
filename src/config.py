import os

VIDEO_FOLDER = 'videos'
DATA_FILE = 'data.xlsx'
N_FRAMES = 30  # Number of frames to sample from each video
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"






import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
SHAPE_PREDICTOR_PATH = os.path.join(DATA_DIR, 'shape_predictor_68_face_landmarks.dat')

# Model parameters
N_FRAMES = 30
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Feature extraction
VISUAL_FEATURE_SIZE = 512
FACIAL_FEATURE_SIZE = 136
AUDIO_FEATURE_SIZE = 54
MOTION_FEATURE_SIZE = N_FRAMES - 1
INTERACTION_FEATURE_SIZE = 1
TEMPORAL_FEATURE_SIZE = 3

FEATURE_TYPES = [
    ('Visual', VISUAL_FEATURE_SIZE),
    ('Facial', FACIAL_FEATURE_SIZE),
    ('Audio', AUDIO_FEATURE_SIZE),
    ('Motion', MOTION_FEATURE_SIZE),
    ('Interaction', INTERACTION_FEATURE_SIZE),
    ('Temporal', TEMPORAL_FEATURE_SIZE)
]