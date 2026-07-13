"""
Test a trained model against a labeled folder of photos / videos.
======================================================================

Point this at a folder structured ONE SUBFOLDER PER LETTER:

    data/test_set/
        A/  some_photo.jpg   a_clip.mp4 ...
        B/  ...
        Ș/  ...

The subfolder NAME is the ground-truth label. Every image/video inside is
fed through the SAME 216-feature pipeline used in training
(data_preparation/feature_extractor.py) and the trained model
(models/alphabet/best_model.pth) predicts a letter. We then report:

  - per-file prediction (correct / wrong + confidence)
  - per-letter accuracy
  - overall accuracy
  - the most common confusions

Usage:
    python evaluation/test_folder.py                     # uses data/test_set
    python evaluation/test_folder.py path/to/my_test_set
    python evaluation/test_folder.py path/to/set --model models/alphabet/best_model.pth

This is INFERENCE ONLY. It never touches your training data or the model
weights, so it is safe to run as often as you like.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

_BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(_BASE))

from data_preparation.feature_extractor import (
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    TOTAL_FEATURES_SIZE,
    resample_frame_indices,
)
from training.train_model import (
    CNNBiLSTMAttnModel,
    CNNLSTMModel,
    LSTMModel,
    TransformerModel,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


# ---------------------------------------------------------------------------
# Config (mirror config.py when available, else fall back to dataset defaults)
# ---------------------------------------------------------------------------

def _load_cfg():
    try:
        import config as cfg
        return {
            'sequence_length': cfg.SEQUENCE_LENGTH,
            'target_fps': cfg.TARGET_FPS,
            'extract_face_mesh': cfg.EXTRACT_FACE_MESH,
            'extract_pose': cfg.EXTRACT_POSE,
            'face_mesh_anchors': cfg.FACE_MESH_ANCHORS,
            'min_detection_confidence': cfg.MIN_DETECTION_CONFIDENCE,
            'hand_dropout_bridge_frames': cfg.HAND_DROPOUT_BRIDGE_FRAMES,
            'model_type': cfg.MODEL_TYPE,
            'dropout': cfg.DROPOUT,
            'model_dir': Path(cfg.MODEL_DIR),
        }
    except Exception:
        return {
            'sequence_length': 45,
            'target_fps': 30,
            'extract_face_mesh': True,
            'extract_pose': True,
            'face_mesh_anchors': DEFAULT_FACE_ANCHORS,
            'min_detection_confidence': 0.5,
            'hand_dropout_bridge_frames': 3,
            'model_type': 'cnn_bilstm_attn',
            'dropout': 0.3,
            'model_dir': _BASE / 'models' / 'alphabet',
        }


# ---------------------------------------------------------------------------
# Feature extraction for a single image / video -> list of 45-frame windows
# ---------------------------------------------------------------------------

def _extract_image_sequence(path, cfg):
    """One image -> single (T, 216) window (the static frame tiled T times)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size > 0 else None
    if bgr is None:
        return None, 'could not decode image'
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=cfg['min_detection_confidence'])
    face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=cfg['min_detection_confidence']) if cfg['extract_face_mesh'] else None
    pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1,
        min_detection_confidence=cfg['min_detection_confidence']) if cfg['extract_pose'] else None
    try:
        hand_res = hands.process(rgb)
        if not (hand_res and hand_res.multi_hand_landmarks):
            return None, 'no hand detected'
        face_res = face.process(rgb) if face else None
        pose_res = pose.process(rgb) if pose else None

        ffe = FrameFeatureExtractor(
            face_anchors=cfg['face_mesh_anchors'],
            bridge_frames=cfg['hand_dropout_bridge_frames'],
            use_face_mesh=cfg['extract_face_mesh'])
        feat = ffe.process_frame(
            hand_landmarks_list=hand_res.multi_hand_landmarks,
            handedness_list=hand_res.multi_handedness,
            face_landmarks=face_res.multi_face_landmarks[0] if (face_res and face_res.multi_face_landmarks) else None,
            pose_landmarks=pose_res.pose_landmarks if (pose_res and pose_res.pose_landmarks) else None,
        )
    finally:
        hands.close()
        if face: face.close()
        if pose: pose.close()

    T = cfg['sequence_length']
    window = np.tile(feat[np.newaxis, :], (T, 1)).astype(np.float32)
    return [window], None


def _extract_video_sequences(path, cfg):
    """One video -> list of (T, 216) sliding windows over the frame stream."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 'could not open video'
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg['target_fps']

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        return None, 'no frames'

    idx = resample_frame_indices(len(frames), src_fps, cfg['target_fps'])

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=cfg['min_detection_confidence'],
        min_tracking_confidence=cfg['min_detection_confidence'])
    face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=cfg['min_detection_confidence']) if cfg['extract_face_mesh'] else None
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=cfg['min_detection_confidence']) if cfg['extract_pose'] else None

    ffe = FrameFeatureExtractor(
        face_anchors=cfg['face_mesh_anchors'],
        bridge_frames=cfg['hand_dropout_bridge_frames'],
        use_face_mesh=cfg['extract_face_mesh'])

    feats = []
    try:
        for i in idx:
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            face_res = face.process(rgb) if face else None
            pose_res = pose.process(rgb) if pose else None
            feat = ffe.process_frame(
                hand_landmarks_list=hand_res.multi_hand_landmarks if hand_res else None,
                handedness_list=hand_res.multi_handedness if hand_res else None,
                face_landmarks=face_res.multi_face_landmarks[0] if (face_res and face_res.multi_face_landmarks) else None,
                pose_landmarks=pose_res.pose_landmarks if (pose_res and pose_res.pose_landmarks) else None,
            )
            feats.append(feat)
    finally:
        hands.close()
        if face: face.close()
        if pose: pose.close()

    feats = np.asarray(feats, dtype=np.float32)
    if (feats[:, -6:-4] > 0).any(axis=1).sum() == 0:
        return None, 'no hand detected in any frame'

    T = cfg['sequence_length']
    if len(feats) < T:
        pad = np.tile(feats[-1:], (T - len(feats), 1))
        feats = np.concatenate([feats, pad], axis=0)

    # Sliding windows (stride ~ half the window).
    stride = max(1, T // 2)
    windows = []
    for start in range(0, len(feats) - T + 1, stride):
        windows.append(feats[start:start + T])
    if not windows:
        windows.append(feats[:T])
    return windows, None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(cfg, num_classes):
    mt = cfg['model_type']
    if mt == 'cnn_bilstm_attn':
        return CNNBiLSTMAttnModel(TOTAL_FEATURES_SIZE, num_classes, dropout=cfg['dropout'])
    if mt == 'cnn_lstm':
        return CNNLSTMModel(TOTAL_FEATURES_SIZE, num_classes, dropout=cfg['dropout'])
    if mt == 'transformer':
        return TransformerModel(TOTAL_FEATURES_SIZE, num_classes, dropout=cfg['dropout'])
    return LSTMModel(TOTAL_FEATURES_SIZE, num_classes=num_classes, dropout=cfg['dropout'])


def _predict(model, windows, device):
    """Average softmax probabilities over all windows of a clip."""
    x = torch.from_numpy(np.stack(windows)).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).mean(dim=0)
    conf, idx = torch.max(probs, dim=0)
    return int(idx.item()), float(conf.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Test a model against a labeled photo/video folder.')
    ap.add_argument('test_dir', nargs='?', default=str(_BASE / 'data' / 'test_set'),
                    help='Folder with one subfolder per letter (default: data/test_set)')
    ap.add_argument('--model', default=None, help='Path to .pth (default: <MODEL_DIR>/best_model.pth)')
    args = ap.parse_args()

    cfg = _load_cfg()
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        print(f'Test folder not found: {test_dir}')
        print('Create it with one subfolder per letter, e.g. data/test_set/A/photo.jpg')
        return

    labels_path = cfg['model_dir'] / 'class_labels.json'
    with open(labels_path, encoding='utf-8') as f:
        class_names = json.load(f)['classes']
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    model_path = Path(args.model) if args.model else cfg['model_dir'] / 'best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print('=' * 64)
    print(f'Model:   {model_path}')
    print(f'Test:    {test_dir}')
    print(f'Classes: {len(class_names)}   Device: {device}')
    print('=' * 64)

    letter_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    per_letter = defaultdict(lambda: {'correct': 0, 'total': 0})
    confusions = Counter()
    skipped = []
    total_correct = total = 0

    for ld in letter_dirs:
        true_label = ld.name
        if true_label not in name_to_idx:
            print(f'\n[skip] "{true_label}" is not a class the model knows: {class_names}')
            continue
        files = [p for p in sorted(ld.iterdir())
                 if p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in VIDEO_EXTS]
        if not files:
            continue
        print(f'\n--- {true_label} ({len(files)} files) ---')
        for fp in files:
            if fp.suffix.lower() in IMAGE_EXTS:
                windows, err = _extract_image_sequence(fp, cfg)
            else:
                windows, err = _extract_video_sequences(fp, cfg)
            if windows is None:
                print(f'  [skip] {fp.name}: {err}')
                skipped.append((str(fp), err))
                continue

            pred_idx, conf = _predict(model, windows, device)
            pred_label = class_names[pred_idx]
            ok = (pred_label == true_label)
            per_letter[true_label]['total'] += 1
            per_letter[true_label]['correct'] += int(ok)
            total += 1
            total_correct += int(ok)
            if not ok:
                confusions[(true_label, pred_label)] += 1
            mark = 'OK ' if ok else 'XX '
            print(f'  {mark} {fp.name:<32} -> {pred_label}  ({conf*100:.1f}%)')

    # --- Summary ---
    print('\n' + '=' * 64)
    print('PER-LETTER ACCURACY')
    print('=' * 64)
    for letter in sorted(per_letter.keys()):
        c, t = per_letter[letter]['correct'], per_letter[letter]['total']
        bar = '#' * int(round(20 * c / t)) if t else ''
        print(f'  {letter:<3} {c:>3}/{t:<3}  {100*c/t:5.1f}%  {bar}')

    print('\n' + '=' * 64)
    if total:
        print(f'OVERALL: {total_correct}/{total} = {100*total_correct/total:.1f}% correct')
    else:
        print('No testable files found.')
    if confusions:
        print('\nTop confusions (true -> predicted):')
        for (t_lbl, p_lbl), n in confusions.most_common(10):
            print(f'  {t_lbl} -> {p_lbl}: {n}')
    if skipped:
        print(f'\nSkipped {len(skipped)} files (no hand / unreadable).')
    print('=' * 64)


if __name__ == '__main__':
    main()
    import os
    os._exit(0)
