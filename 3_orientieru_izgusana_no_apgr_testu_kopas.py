import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

#iestatijumi testu kopai
DATASET_DIR = "testa_datu_kopa_trimmed"
OUTPUT_DIR  = "dataset_landmarks_test"
MODEL_PATH  = os.path.join("models", "hand_landmarker.task")

#mapju nosaukumi mapītē testa_datu_kopa_trimmed
CLASS_FOLDERS = [
    "A",
    "AA_long",
    "ata",
    "B",
    "labdien",
    "ne",
    "ok",
    "paldies",
    "plane",
    "U",
    "es",
    "vajag",
    "nauda",
    "tualete",
    "telefons",
]

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")

MAX_NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.35
MIN_HAND_PRESENCE_CONFIDENCE = 0.35
MIN_TRACKING_CONFIDENCE = 0.35

SAVE_METADATA_CSV = True

#paligfunkcijas aka helpers, kas palidz galvenajam kodam
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def folder_to_label(folder_name: str) -> str:
    """
    Testa mapēm _new nav, tāpēc atgriež tieši mapes nosaukumu.
    """
    if folder_name.endswith("_new"):
        return folder_name[:-4]
    return folder_name

def create_hand_landmarker():
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)

def detect_landmarks_in_frame(landmarker, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    hand = result.hand_landmarks[0]
    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
    return arr

def process_video(video_path: str, landmarker):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None, 0, 0

    total_frames = 0
    detected_frames = 0
    sequence = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        landmarks = detect_landmarks_in_frame(landmarker, frame)
        if landmarks is not None:
            sequence.append(landmarks)
            detected_frames += 1

    cap.release()

    if detected_frames == 0:
        return None, total_frames, detected_frames

    sequence = np.stack(sequence, axis=0)
    return sequence, total_frames, detected_frames

def save_npz(output_path: str, sequence: np.ndarray, label: str,
             video_path: str, total_frames: int, detected_frames: int):
    np.savez_compressed(
        output_path,
        sequence=sequence,
        label=label,
        video_path=video_path,
        total_frames=total_frames,
        detected_frames=detected_frames,
    )

def classify_quality(detection_ratio: float, detected_frames: int):
    if detected_frames == 0:
        return "no_landmarks"
    if detection_ratio >= 0.90:
        return "good"
    if detection_ratio >= 0.70:
        return "medium"
    return "low"

#main
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ensure_dir(OUTPUT_DIR)

    label_names = [folder_to_label(folder) for folder in CLASS_FOLDERS]
    for label in sorted(set(label_names)):
        ensure_dir(os.path.join(OUTPUT_DIR, label))

    metadata_rows = []
    summary_rows = []

    landmarker = create_hand_landmarker()

    try:
        for folder_name in CLASS_FOLDERS:
            input_dir = os.path.join(DATASET_DIR, folder_name)
            label = folder_to_label(folder_name)
            output_dir = os.path.join(OUTPUT_DIR, label)

            if not os.path.isdir(input_dir):
                print(f"[WARNING] Missing folder: {input_dir}")
                continue

            video_files = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]
            video_files.sort()

            print(f"\n=== Processing folder: {folder_name} -> label: {label} | videos: {len(video_files)} ===")

            class_total = 0
            class_saved = 0
            class_good = 0
            class_medium = 0
            class_low = 0
            class_no_landmarks = 0

            for video_name in video_files:
                class_total += 1

                video_path = os.path.join(input_dir, video_name)
                base_name = os.path.splitext(video_name)[0]
                output_path = os.path.join(output_dir, f"{base_name}.npz")

                sequence, total_frames, detected_frames = process_video(video_path, landmarker)
                detection_ratio = detected_frames / total_frames if total_frames > 0 else 0.0
                quality = classify_quality(detection_ratio, detected_frames)

                if sequence is None:
                    print(
                        f"[SKIP] {folder_name}/{video_name} | "
                        f"detected={detected_frames}/{total_frames} ({detection_ratio:.2%}) | "
                        f"status=no_landmarks"
                    )
                    class_no_landmarks += 1
                    metadata_rows.append([
                        label, folder_name, video_name, "",
                        total_frames, detected_frames,
                        f"{detection_ratio:.4f}", "no_landmarks"
                    ])
                    continue

                save_npz(output_path, sequence, label, video_path, total_frames, detected_frames)

                class_saved += 1
                if quality == "good": class_good += 1
                elif quality == "medium": class_medium += 1
                else: class_low += 1

                print(
                    f"[OK] {folder_name}/{video_name} -> {base_name}.npz | "
                    f"seq_shape={sequence.shape} | "
                    f"detected={detected_frames}/{total_frames} ({detection_ratio:.2%}) | "
                    f"status={quality}"
                )

                metadata_rows.append([
                    label, folder_name, video_name, f"{base_name}.npz",
                    total_frames, detected_frames,
                    f"{detection_ratio:.4f}", quality
                ])

            summary_rows.append([
                label, folder_name, class_total, class_saved,
                class_good, class_medium, class_low, class_no_landmarks
            ])

    finally:
        landmarker.close()

    if SAVE_METADATA_CSV:
        metadata_csv = os.path.join(OUTPUT_DIR, "landmarks_metadata.csv")
        summary_csv = os.path.join(OUTPUT_DIR, "landmarks_summary.csv")

        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label","source_folder","video_name","npz_name",
                             "total_frames","detected_frames","detection_ratio","status"])
            writer.writerows(metadata_rows)

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label","source_folder","total_videos","saved_npz",
                             "good","medium","low","no_landmarks"])
            writer.writerows(summary_rows)

        print(f"\nSaved metadata CSV: {metadata_csv}")
        print(f"Saved summary CSV: {summary_csv}")

    print("\nDone.")

if __name__ == "__main__":
    main()