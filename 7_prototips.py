import os
import json
import time
from collections import deque, Counter
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
#iestatijumi
MODEL_DIR = "training_outputs_bilstm_group_split_15classes"
MODEL_PATH = os.path.join(MODEL_DIR, "best_bilstm_group_split.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map_group_split.json")
HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")
SEQUENCE_LENGTH = 48
CONF_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.30
MIN_HAND_PRESENCE_CONFIDENCE = 0.30
MIN_TRACKING_CONFIDENCE = 0.30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_EVERY_N_FRAMES = 2
MIN_FRAMES_FOR_PREDICTION = 8
MAX_MISSED_FRAMES = 8
DISPLAY_HOLD_FRAMES = 10

#same modelis no apmacibas
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.classifier(out)
        return out

#Hand Landmarker failins
def create_hand_landmarker():
    if not os.path.exists(HAND_MODEL_PATH):
        raise FileNotFoundError(f"Hand landmarker model not found: {HAND_MODEL_PATH}")

    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return vision.HandLandmarker.create_from_options(options)

#atrast roku kadra
def detect_landmarks_in_frame(landmarker, frame_bgr, timestamp_ms):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    hand = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

#normalizacija pret plaukstas 0. punktu
def normalize_sequence_oldnorm(seq):
    wrist = seq[:, 0:1, :]
    seq = seq - wrist

    scale = np.max(np.linalg.norm(seq, axis=2))
    if scale < 1e-6:
        scale = 1.0

    seq = seq / scale
    return seq.astype(np.float32)

#sekvences parveidosana uz 48 kadriem
def resample_sequence(seq, target_len=48):
    T = len(seq)

    if T == target_len:
        return seq

    if T == 1:
        return np.repeat(seq, target_len, axis=0)

    idxs = np.linspace(0, T - 1, target_len)
    idxs = np.round(idxs).astype(np.int32)
    return seq[idxs]

#prieksapstrade
def preprocess_sequence(seq):
    seq = normalize_sequence_oldnorm(seq)
    seq = resample_sequence(seq, SEQUENCE_LENGTH)
    seq = seq.reshape(SEQUENCE_LENGTH, -1).astype(np.float32)
    return seq

#prognozes izlidzinasana
def get_smoothed_prediction(pred_buffer):
    if len(pred_buffer) == 0:
        return None

    counts = Counter(pred_buffer)
    return counts.most_common(1)[0][0]

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

#load klasu karti
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

index_to_label = {int(k): v for k, v in label_map["index_to_label"].items()}
num_classes = len(index_to_label)

print("Loaded labels:")
for idx, label in index_to_label.items():
    print(f"  {idx}: {label}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = BiLSTMClassifier(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"\nModel loaded on {DEVICE}")
print(f"Using model: {MODEL_PATH}")
print(f"Using label map: {LABEL_MAP_PATH}")
print("Using normalization: oldnorm")

landmarker = create_hand_landmarker()

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

display_label = ""
display_conf = 0.0

missed_frames = 0
hold_frames_left = 0
frame_counter = 0

fps_timer = time.time()
fps_counter = 0
fps_value = 0.0

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Nevar atvērt kameru")

WINDOW_NAME = "Prototips demonstresana"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

start_time = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_counter += 1
        fps_counter += 1

        now = time.time()
        if now - fps_timer >= 1.0:
            fps_value = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer = now

        timestamp_ms = int((now - start_time) * 1000)

        landmarks = detect_landmarks_in_frame(landmarker, frame, timestamp_ms)
        hand_detected = landmarks is not None

        raw_pred_label = ""
        raw_pred_conf = 0.0

        if hand_detected:
            missed_frames = 0
            sequence_buffer.append(landmarks)

            if len(sequence_buffer) >= MIN_FRAMES_FOR_PREDICTION and frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                seq = np.array(sequence_buffer, dtype=np.float32)
                features = preprocess_sequence(seq)

                tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                pred_label = index_to_label[pred_idx]

                raw_pred_label = pred_label
                raw_pred_conf = pred_conf

                if pred_conf >= CONF_THRESHOLD:
                    prediction_buffer.append(pred_idx)

                if len(prediction_buffer) > 0:
                    smooth_idx = get_smoothed_prediction(prediction_buffer)
                    display_label = index_to_label[smooth_idx]
                    display_conf = pred_conf
                    hold_frames_left = DISPLAY_HOLD_FRAMES

        else:
            missed_frames += 1

            if missed_frames >= MAX_MISSED_FRAMES:
                sequence_buffer.clear()
                prediction_buffer.clear()

                if hold_frames_left <= 0:
                    display_label = ""
                    display_conf = 0.0

        if hold_frames_left > 0:
            hold_frames_left -= 1

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 95), (30, 30, 30), -1)

        if display_label:
            cv2.putText(frame, f"{display_label} ({display_conf:.2f})", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)

        if raw_pred_label:
            cv2.putText(frame, f"raw: {raw_pred_label} ({raw_pred_conf:.2f})", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        if not hand_detected and not display_label:
            cv2.putText(frame, "Nav rokas", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)

        cv2.putText(frame, f"buffer: {len(sequence_buffer)}", (15, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"missed: {missed_frames}", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"fps: {fps_value:.1f}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
