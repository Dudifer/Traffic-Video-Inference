from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import tempfile
import shutil
import pysqlite3 as sql
import cv2
from gui.model import runOnVideo, runOnVideoStream

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    # Run YOLO tracking over the full video, saving annotated output
    annotated_path = tempfile.mktemp(suffix="_annotated.mp4")
    track_log, frame_count = runOnVideo(temp_path, output_path=annotated_path)

    # Persist results to SQLite
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
        "annotated_video": annotated_path,
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


@app.post("/stream")
async def stream(file: UploadFile = File(...)):
    """Stream annotated frames as MJPEG back to the local machine."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    def generate():
        for frame in runOnVideoStream(temp_path):
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/download")
def download(path: str):
    """Serve an annotated video file back to the local machine."""
    return FileResponse(path, media_type="video/mp4", filename="annotated_output.mp4")