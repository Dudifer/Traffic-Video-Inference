from ultralytics import YOLO
import cv2
from typing import Generator


def _load_model():
    return YOLO("yolov8n.pt")


def runOnVideo(video_path: str, output_path: str = "annotated_output.mp4") -> tuple[dict, int, dict]:
    """
    Run YOLO tracking over the full video in one call.
    Saves an annotated copy of the video with bounding boxes drawn.
    Returns (track_log, frame_count, class_names).
    """
    model = _load_model()
    class_names = model.names  # {0: 'person', 2: 'car', 5: 'bus', 7: 'truck', ...}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=.65
    )

    track_log = {}
    processed = 0

    for frame_idx, r in enumerate(results):
        now = frame_idx / fps

        annotated_frame = r.plot()
        writer.write(annotated_frame)

        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                track_id = int(box.id.item()) if box.id is not None else None
                class_id = int(box.cls.item()) if box.cls is not None else None
                confidence = float(box.conf.item()) if box.conf is not None else None

                if track_id is None:
                    continue

                if track_id not in track_log:
                    track_log[track_id] = {
                        "class_id": class_id,
                        "class_name": class_names.get(class_id, "unknown"),
                        "confidence": confidence,
                        "first_seen": now,
                        "last_seen": now
                    }
                else:
                    track_log[track_id]["last_seen"] = now
                    track_log[track_id]["confidence"] = confidence

        processed += 1

    writer.release()
    return track_log, processed, class_names


def runOnVideoStream(video_path: str) -> Generator:
    """
    Run YOLO tracking and yield (annotated_frame, track_log, class_names) per frame.
    track_log is updated in place — the final yield reflects the complete session.
    """
    model = _load_model()
    class_names = model.names

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.55
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    track_log = {}

    for frame_idx, r in enumerate(results):
        now = frame_idx / fps

        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                track_id = int(box.id.item()) if box.id is not None else None
                class_id = int(box.cls.item()) if box.cls is not None else None
                confidence = float(box.conf.item()) if box.conf is not None else None

                if track_id is None:
                    continue

                if track_id not in track_log:
                    track_log[track_id] = {
                        "class_id": class_id,
                        "class_name": class_names.get(class_id, "unknown"),
                        "confidence": confidence,
                        "first_seen": now,
                        "last_seen": now
                    }
                else:
                    track_log[track_id]["last_seen"] = now
                    track_log[track_id]["confidence"] = confidence

        yield r.plot(), track_log, class_names