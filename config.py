"""
Configuration file for gesture recognition pipeline.
Edit these settings instead of modifying the main scripts.
"""

# ============================================================================
# DATA EXTRACTION SETTINGS
# ============================================================================

# Directory containing your training videos
# Expected structure: VIDEO_DIR/gesture_name/video1.mp4
from pathlib import Path as _Path
_BASE = _Path(__file__).parent

VIDEO_DIR = str(_BASE / "alphabet")

# Directory to save processed data
OUTPUT_DIR = str(_BASE / "data" / "alphabet_processed")

# Number of consecutive frames per sequence
# - Short gestures (snaps, clicks): 15-20
# - Medium gestures (waves, swipes): 30-40
# - Long gestures (sign language): 50-60
# Bumped to 45 so moving signs (J, Z, Q, Ș, Ț) and future verbs/words have
# enough temporal context (~1.5s at 30fps effective rate).
SEQUENCE_LENGTH = 45

# Effective frame rate used for both training and inference. Source videos
# (any FPS) and the live camera (60fps) are resampled to this rate so a
# sequence of SEQUENCE_LENGTH always represents the same wall-clock duration.
TARGET_FPS = 30

# MediaPipe detection confidence (0.0 to 1.0)
# Lower = more detections but more false positives
# Lowered to 0.5 for better hand detection in sign language videos
MIN_DETECTION_CONFIDENCE = 0.5

# MediaPipe tracking confidence (0.0 to 1.0)
MIN_TRACKING_CONFIDENCE = 0.5

# Extract both hands or just one
EXTRACT_BOTH_HANDS = True

# Pose landmarks (shoulders, hips) used as a stable body-anchor frame so
# the global wrist trajectory survives across frames. REQUIRED for moving
# signs.
EXTRACT_POSE = True

# Extract face mesh landmarks (for face-related signs)
EXTRACT_FACE_MESH = True

# Face mesh anchor indices (key points for sign language).
# 20 anchors x 3 coords = 60 features. This MUST stay in sync with
# data_preparation/feature_extractor.DEFAULT_FACE_ANCHORS.
FACE_MESH_ANCHORS = {
    'nose_tip': 1,
    'nose_bridge': 6,
    'left_eye_inner': 133,
    'left_eye_outer': 33,
    'left_eye_center': 468,  # iris center (fallback to 159)
    'right_eye_inner': 362,
    'right_eye_outer': 263,
    'right_eye_center': 473,  # iris center (fallback to 386)
    'left_eyebrow_inner': 107,
    'left_eyebrow_outer': 70,
    'right_eyebrow_inner': 336,
    'right_eyebrow_outer': 300,
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'chin': 152,
    'forehead': 10,
    'left_cheek': 123,
    'right_cheek': 352,
}

# Number of face anchor points (20 key points × 3 coordinates = 60 features)
NUM_FACE_ANCHORS = len(FACE_MESH_ANCHORS)
FACE_FEATURES_SIZE = NUM_FACE_ANCHORS * 3  # 60 features

# ============================================================================
# FEATURE LAYOUT (per frame, for moving-sign / trajectory recognition)
# ============================================================================
# Block A: per-frame local handshape (wrist-centered, scale-normalized)
#   2 hands × 21 landmarks × 3 = 126
# Block B: face anchors (nose-centered, face-height-normalized)
#   20 anchors × 3 = 60
# Block C: GLOBAL wrist position in body-anchor frame (NOT re-centered per
#          frame). This is the trajectory signal that distinguishes J/Z/Q.
#   2 hands × 3 = 6
# Block D: GLOBAL index-fingertip position in body-anchor frame.
#   2 hands × 3 = 6
# Block E: wrist velocity (frame-to-frame delta of Block C)
#   2 hands × 3 = 6
# Block F: wrist acceleration (frame-to-frame delta of Block E)
#   2 hands × 3 = 6
# Block G: missing-mask + reserved padding (kept zero unless present)
#   2 hands present-flags + 4 reserved slots = 6
# ----------------------------------------------------------------------------
# TOTAL = 126 + 60 + 6 + 6 + 6 + 6 + 6 = 216
HAND_LOCAL_SIZE = 126
WRIST_GLOBAL_SIZE = 6
FINGERTIP_GLOBAL_SIZE = 6
WRIST_VELOCITY_SIZE = 6
WRIST_ACCEL_SIZE = 6
MASK_RESERVED_SIZE = 6

TOTAL_FEATURES_SIZE = (
    HAND_LOCAL_SIZE
    + (FACE_FEATURES_SIZE if EXTRACT_FACE_MESH else 0)
    + WRIST_GLOBAL_SIZE
    + FINGERTIP_GLOBAL_SIZE
    + WRIST_VELOCITY_SIZE
    + WRIST_ACCEL_SIZE
    + MASK_RESERVED_SIZE
)

# Hand-dropout bridging: if MediaPipe loses a hand for <= this many
# consecutive frames, we hold the last known landmarks instead of zero-padding.
# Beyond this, we mark the hand as missing.
HAND_DROPOUT_BRIDGE_FRAMES = 3

# Minimum fraction of frames in a sequence that must contain at least one
# detected hand. Low-quality sequences are dropped at extraction time.
MIN_HAND_PRESENCE_RATIO = 0.5

# Letters whose meaning depends on the direction of motion (do NOT augment
# with horizontal flip / time-reverse).
DIRECTIONAL_LETTERS = {'J', 'Z', 'Q', 'Ș', 'Ț', 'S', 'Y'}

# Letters that ARE directional (above) but ALSO have a valid static form in
# the signed alphabet. JPG/PNG photos in these letter folders WILL be used by
# the static-image extractor; videos still feed the motion variant. Both
# representations train the same class.
DIRECTIONAL_LETTERS_WITH_STATIC_FORM = {'Z'}

# Normalize coordinates relative to hand size
NORMALIZE_COORDINATES = True

# Extract additional features (distances, angles, etc.)
EXTRACT_FEATURES = True

# Process every Nth frame (1 = all frames, 2 = every other frame)
# Higher values = faster processing but less data
SAMPLE_RATE = 1

# Show visualization while processing
VISUALIZE_EXTRACTION = False

# ============================================================================
# DATA SPLITTING
# ============================================================================

# Train/validation/test split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Optional: Map folder names to custom labels
# Example: {'folder_name': 'custom_label'}
GESTURE_MAPPING = {
    # 'thumbs_up_videos': 'thumbs_up',
    # 'peace_sign_vids': 'peace',
}

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Directory to save trained models
MODEL_DIR = str(_BASE / "models" / "alphabet")

# Model architecture
# Options: 'lstm', 'cnn_lstm', 'cnn_bilstm_attn', 'transformer'
# - 'lstm': Fast, good for simple gestures
# - 'cnn_lstm': Old default
# - 'cnn_bilstm_attn': Bidirectional LSTM with attention pooling (RECOMMENDED
#   for moving signs and word sequences — the new default).
# - 'transformer': Best accuracy, needs more data
MODEL_TYPE = 'cnn_bilstm_attn'

# Training parameters
EPOCHS = 150  # More epochs for small alphabet dataset
BATCH_SIZE = 32  # Increase to 64/128 if you have lots of data
LEARNING_RATE = 0.001

# Model architecture hyperparameters
LSTM_UNITS = [128, 64]  # Number of units in each LSTM layer
DROPOUT = 0.3  # Dropout for regularization (raised from 0.2 to fight overfitting)

# Use class-weighted CrossEntropy (inverse frequency) so under-represented
# moving letters (J, Z, Q, Ș, Ț) get full gradient signal.
USE_CLASS_WEIGHTS = True

# Gradient clipping threshold (helps stability with attention pooling).
GRAD_CLIP_NORM = 1.0

# AdamW weight decay (decoupled L2 regularization).
WEIGHT_DECAY = 1e-4

# Label smoothing for CrossEntropy: prevents the model from becoming
# over-confident and improves probability calibration.
LABEL_SMOOTHING = 0.1

# Transformer-specific settings (only used if MODEL_TYPE='transformer')
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 128

# ============================================================================
# CALLBACK SETTINGS
# ============================================================================

# Early stopping patience (stop if no improvement for N epochs)
EARLY_STOPPING_PATIENCE = 25

# Learning rate reduction patience
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.5

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

# Minimum confidence threshold for real-time predictions
PREDICTION_CONFIDENCE_THRESHOLD = 0.7

# Smoothing window for real-time predictions (number of frames to average)
PREDICTION_SMOOTHING_WINDOW = 3

# ============================================================================
# HARDWARE SETTINGS
# ============================================================================

# Enable GPU memory growth (prevents OOM errors)
ENABLE_GPU_MEMORY_GROWTH = True

# Mixed precision training (faster on RTX 4080)
# Requires TensorFlow 2.4+ or PyTorch 1.6+
ENABLE_MIXED_PRECISION = False  # Set to True for faster training

# Number of CPU threads for data loading
NUM_CPU_THREADS = 4

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Data augmentation (experimental)
ENABLE_AUGMENTATION = False

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'time_stretch_factor': 0.1,  # +/- 10% time stretching
    'random_frame_drop_rate': 0.05,  # Drop 5% of frames randomly
    'add_noise_std': 0.01,  # Add small Gaussian noise
}

# Save checkpoints every N epochs
CHECKPOINT_FREQUENCY = 5

# Verbose training output
VERBOSE_TRAINING = 1  # 0=silent, 1=progress bar, 2=one line per epoch


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_extraction_config():
    """Get configuration for data extraction."""
    return {
        'sequence_length': SEQUENCE_LENGTH,
        'target_fps': TARGET_FPS,
        'min_detection_confidence': MIN_DETECTION_CONFIDENCE,
        'min_tracking_confidence': MIN_TRACKING_CONFIDENCE,
        'extract_both_hands': EXTRACT_BOTH_HANDS,
        'extract_pose': EXTRACT_POSE,
        'extract_face_mesh': EXTRACT_FACE_MESH,
        'face_mesh_anchors': FACE_MESH_ANCHORS,
        'normalize': NORMALIZE_COORDINATES,
        'extract_features': EXTRACT_FEATURES,
        'hand_dropout_bridge_frames': HAND_DROPOUT_BRIDGE_FRAMES,
        'min_hand_presence_ratio': MIN_HAND_PRESENCE_RATIO,
        'directional_letters': DIRECTIONAL_LETTERS,
    }


def get_training_config():
    """Get configuration for model training."""
    return {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'lstm_units': LSTM_UNITS,
        'dropout': DROPOUT,
        'model_type': MODEL_TYPE,
    }


def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check split ratios
    total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if abs(total_split - 1.0) > 0.001:
        errors.append(f"Train/Val/Test splits must sum to 1.0 (currently {total_split})")

    # Check confidence thresholds
    if not (0.0 <= MIN_DETECTION_CONFIDENCE <= 1.0):
        errors.append("MIN_DETECTION_CONFIDENCE must be between 0.0 and 1.0")

    if not (0.0 <= MIN_TRACKING_CONFIDENCE <= 1.0):
        errors.append("MIN_TRACKING_CONFIDENCE must be between 0.0 and 1.0")

    # Check sequence length
    if SEQUENCE_LENGTH < 5:
        errors.append("SEQUENCE_LENGTH should be at least 5")

    # Check batch size
    if BATCH_SIZE < 1:
        errors.append("BATCH_SIZE must be at least 1")

    # Check model type
    valid_models = ['lstm', 'cnn_lstm', 'cnn_bilstm_attn', 'transformer']
    if MODEL_TYPE not in valid_models:
        errors.append(f"MODEL_TYPE must be one of {valid_models}")

    if errors:
        print("[ERROR] Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("[OK] Configuration is valid")
    return True


if __name__ == "__main__":
    # Validate configuration when running this file
    print("Validating configuration...")
    validate_config()

    print(f"\nCurrent configuration:")
    print(f"  Data extraction:")
    print(f"    - Video directory: {VIDEO_DIR}")
    print(f"    - Sequence length: {SEQUENCE_LENGTH}")
    print(f"    - Extract both hands: {EXTRACT_BOTH_HANDS}")
    print(f"    - Extract face mesh: {EXTRACT_FACE_MESH}")
    print(f"    - Face anchors: {NUM_FACE_ANCHORS} points ({FACE_FEATURES_SIZE} features)")
    print(f"    - Normalize: {NORMALIZE_COORDINATES}")
    print(f"\n  Feature sizes:")
    print(f"    - Hand local (wrist-centered): {HAND_LOCAL_SIZE}")
    print(f"    - Face anchors: {FACE_FEATURES_SIZE if EXTRACT_FACE_MESH else 0}")
    print(f"    - Wrist global (body-anchored): {WRIST_GLOBAL_SIZE}")
    print(f"    - Fingertip global: {FINGERTIP_GLOBAL_SIZE}")
    print(f"    - Wrist velocity: {WRIST_VELOCITY_SIZE}")
    print(f"    - Wrist acceleration: {WRIST_ACCEL_SIZE}")
    print(f"    - Mask + reserved: {MASK_RESERVED_SIZE}")
    print(f"    - TOTAL per frame: {TOTAL_FEATURES_SIZE}")
    print(f"\n  Model training:")
    print(f"    - Model type: {MODEL_TYPE}")
    print(f"    - Batch size: {BATCH_SIZE}")
    print(f"    - Epochs: {EPOCHS}")
    print(f"    - Learning rate: {LEARNING_RATE}")
