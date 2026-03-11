from fastapi import FastAPI, UploadFile, File
import cv2
import tempfile
import shutil
import numpy as np
import json
import time 
import pysqlite3 as sql
from gui.model import testModel

app = FastAPI()

# tracks first_seen and last_seen timestamps per track_id
track_log = {}
fps = 30

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    track_log = {}
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    results = []

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("End of video reached or failed to read frame.")
            break

        if frame_count % 30 == 0:
            track_log = testModel(frame, frame_count, fps, track_log)

        frame_count += 1

    cap.release()

    conn = sql.connect("detections.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            track_id INTEGER,
            class_id INTEGER,
            confidence REAL,
            first_seen REAL,
            last_seen REAL,
            duration REAL
        )
    """)

    rows = []
    for track_id, data in track_log.items():
        duration = data["last_seen"] - data["first_seen"]
        row = (track_id, data["class_id"], data["confidence"], data["first_seen"], data["last_seen"], duration)
        rows.append(row)
        cursor.execute("INSERT INTO detections VALUES (?,?,?,?,?,?)", row)

    conn.commit()
    conn.close()

    return {
        "frames_processed": frame_count,
        "detections": [
            {
                "track_id": r[0],
                "class_id": r[1],
                "confidence": r[2],
                "first_seen": r[3],
                "last_seen": r[4],
                "duration": r[5]
            } for r in rows
        ]
    }

    return {
        "frames_processed": frame_count,
        "detections": results
    }


#print(response.json())



# uvicorn app:app --reload
# python tkinter_app.py