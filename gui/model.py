from ultralytics import YOLO 
import cv2 

video_path = "C:/Users/Owner/Videos/traffic video.MOV"

def runOneFrame(file_path):
    video = cv2.VideoCapture(file_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    vehicles = {}

    model = YOLO("yolov8n.pt")  # nano model (fast)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, tracker="bytetrack.yaml", persist=True, show = True)
        
        break 
        # if results[0].boxes.id is not None:
        #     boxes = results[0].boxes
        #     for box in results[0].boxes:

# import threading

# def upload_video():
#     file_path = filedialog.askopenfilename(
#         filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
#     )

#     if file_path:
#         threading.Thread(
#             target=run_inference,
#             args=(file_path,)
#         ).start()

def testModel(frame, frame_count, fps, track_log):
    print("running model test")
    model = YOLO("yolov8n.pt")

    results = model.track(
        source=frame,
        tracker="bytetrack.yaml",
        persist=True,
    )

    for r in results:
        boxes = r.boxes
        for box in boxes:
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item()) if box.cls is not None else None
            confidence = float(box.conf.item()) if box.conf is not None else None

            if track_id is None:
                continue

            now = frame_count / fps

            if track_id not in track_log:
                track_log[track_id] = {
                    "class_id": class_id,
                    "confidence": confidence,
                    "first_seen": now,
                    "last_seen": now
                }
            else:
                track_log[track_id]["last_seen"] = now
                track_log[track_id]["confidence"] = confidence

    return track_log


# if __name__ == "__main__":
#     print("starting demo")
#     runOneFrame(video_path)
    #testModel()