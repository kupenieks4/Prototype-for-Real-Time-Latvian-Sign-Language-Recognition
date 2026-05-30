import os
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import cv2


@dataclass
class Config:
    camera_index: int = 0
    mirror: bool = True
    fps: int = 20
    frame_width: int = 1280
    frame_height: int = 720
    record_seconds: int = 4
    countdown_seconds: int = 3
    output_dir: str = "dataset_raw"
    metadata_filename: str = "metadata.csv"
    video_extension: str = ".avi"


class DatasetVideoRecorder:
    def __init__(self, cfg: Config, labels: List[str]):
        self.cfg = cfg
        self.labels = labels
        self.last_saved_path: Optional[str] = None

        os.makedirs(cfg.output_dir, exist_ok=True)

        for label in labels:
            os.makedirs(os.path.join(cfg.output_dir, label), exist_ok=True)

        self.metadata_path = os.path.join(cfg.output_dir, cfg.metadata_filename)
        self.ensure_metadata_file()

    def ensure_metadata_file(self):
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "relative_path",
                    "label",
                    "width",
                    "height",
                    "fps",
                    "record_seconds",
                    "frames",
                    "timestamp"
                ])

    def append_metadata(self, relative_path: str, label: str, frames: int):
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                relative_path,
                label,
                self.cfg.frame_width,
                self.cfg.frame_height,
                self.cfg.fps,
                self.cfg.record_seconds,
                frames,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def remove_last_metadata_row(self, relative_path: str):
        if not os.path.exists(self.metadata_path):
            return

        with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            return

        header = rows[0]
        data_rows = rows[1:]

        for i in range(len(data_rows) - 1, -1, -1):
            if data_rows[i] and data_rows[i][0] == relative_path:
                del data_rows[i]
                break

        with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)

    def get_next_filename(self, label: str) -> str:
        label_dir = os.path.join(self.cfg.output_dir, label)
        existing = [
            f for f in os.listdir(label_dir)
            if f.startswith(label + "_") and f.endswith(self.cfg.video_extension)
        ]

        max_index = 0
        for f in existing:
            name = os.path.splitext(f)[0]
            parts = name.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                max_index = max(max_index, int(parts[-1]))

        next_index = max_index + 1
        filename = f"{label}_{next_index:04d}{self.cfg.video_extension}"
        return os.path.join(label_dir, filename)

    def draw_text(self, frame, text, x=20, y=40, color=(255, 255, 255), scale=0.9, thickness=2):
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def draw_main_overlay(self, frame, current_label: str):
        self.draw_text(frame, f"Label: {current_label}", y=40, color=(0, 0, 255), scale=1.0)
        self.draw_text(frame, "R = record   X = delete last   Q = quit", y=85, color=(255, 255, 255), scale=0.8)

        start_y = 130
        for i, label in enumerate(self.labels, start=1):
            if i < 10:
                key_hint = str(i)
            elif i == 10:
                key_hint = "0"
            else:
                key_hint = "?"

            self.draw_text(frame, f"{key_hint} = {label}", y=start_y + (i - 1) * 30, color=(210, 210, 210), scale=0.75)

    def countdown(self, cap, label: str):
        for sec in range(self.cfg.countdown_seconds, 0, -1):
            start = time.time()
            while time.time() - start < 1.0:
                ok, frame = cap.read()
                if not ok:
                    return False

                frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)

                preview = frame.copy()
                self.draw_text(preview, f"Label: {label}", y=40, color=(0, 0, 255), scale=1.0)
                self.draw_text(preview, f"Recording starts in: {sec}", y=85, color=(0, 255, 255), scale=0.9)
                self.draw_text(preview, "Prepare gesture", y=130, color=(255, 255, 255), scale=0.8)

                cv2.imshow("Dataset Video Recorder", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return False

        return True

    def record_video(self, cap, label: str):
        save_path = self.get_next_filename(label)
        relative_path = os.path.relpath(save_path, self.cfg.output_dir)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            self.cfg.fps,
            (self.cfg.frame_width, self.cfg.frame_height)
        )

        start_time = time.time()
        total_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)

            elapsed = time.time() - start_time
            remaining = max(0.0, self.cfg.record_seconds - elapsed)

            writer.write(frame)
            total_frames += 1

            preview = frame.copy()
            self.draw_text(preview, f"REC | {label}", y=40, color=(0, 0, 255), scale=1.0)
            self.draw_text(preview, f"Remaining: {remaining:.1f}s", y=85, color=(255, 255, 255), scale=0.8)
            self.draw_text(preview, f"Frames: {total_frames}", y=130, color=(210, 210, 210), scale=0.75)

            cv2.imshow("Dataset Video Recorder", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                writer.release()
                return False

            if elapsed >= self.cfg.record_seconds:
                break

        writer.release()

        self.append_metadata(relative_path, label, total_frames)
        self.last_saved_path = save_path

        print(f"[SAVED] {save_path}")
        print(f"[METADATA] added: {relative_path}")
        return True

    def delete_last_recording(self):
        if not self.last_saved_path:
            print("[DELETE] No recording to delete.")
            return

        if not os.path.exists(self.last_saved_path):
            print("[DELETE] Last saved file no longer exists.")
            self.last_saved_path = None
            return

        relative_path = os.path.relpath(self.last_saved_path, self.cfg.output_dir)
        os.remove(self.last_saved_path)
        self.remove_last_metadata_row(relative_path)

        print(f"[DELETE] Removed file: {self.last_saved_path}")
        print(f"[DELETE] Removed metadata row: {relative_path}")

        self.last_saved_path = None

    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.cfg.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        current_label = self.labels[0] if self.labels else "gesture"

        print("Controls:")
        print("  q = quit")
        print("  r = countdown + record")
        print("  x = delete last recording")
        for i, label in enumerate(self.labels, start=1):
            if i < 10:
                key_hint = str(i)
            elif i == 10:
                key_hint = "0"
            else:
                key_hint = "?"
            print(f"  {key_hint} = label {label}")
        print()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)

                preview = frame.copy()
                self.draw_main_overlay(preview, current_label)

                cv2.imshow("Dataset Video Recorder", preview)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("r"):
                    ok_countdown = self.countdown(cap, current_label)
                    if not ok_countdown:
                        break
                    ok_record = self.record_video(cap, current_label)
                    if not ok_record:
                        break
                elif key == ord("x"):
                    self.delete_last_recording()
                else:
                    for i, label in enumerate(self.labels, start=1):
                        if i < 10 and key == ord(str(i)):
                            current_label = label
                            print(f"[LABEL] {label}")
                            break
                        elif i == 10 and key == ord("0"):
                            current_label = label
                            print(f"[LABEL] {label}")
                            break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    cfg = Config(
        camera_index=0,
        mirror=True,
        fps=20,
        frame_width=1280,
        frame_height=720,
        record_seconds=4,
        countdown_seconds=3,
        output_dir="dataset_raw",
        metadata_filename="metadata.csv",
        video_extension=".avi"
    )

    labels = [
        "A",
        "U",
        "AA_long",
        "plane",
        "labdien",
        "ata",
        "ok",
        "ne",
        "B",
        "paldies"
    ]

    app = DatasetVideoRecorder(cfg, labels)
    app.run()


if __name__ == "__main__":
    main()