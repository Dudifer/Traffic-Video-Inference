from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import tempfile
import shutil
import uuid
import pysqlite3 as sql
import cv2
from gui.model import runOnVideo, runOnVideoStream

app = FastAPI()

# In-memory cache of results from /stream sessions, keyed by session_id
stream_results: dict = {}


def save_to_db(track_log: dict, frame_count: int) -> list:
    """Persist track_log to SQLite and return rows."""
    conn = sql.connect("detections.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            track_id INTEGER,
            class_id INTEGER,
            class_name TEXT,
            confidence REAL,
            first_seen REAL,
            last_seen REAL,
            duration REAL
        )
    """)

    rows = []
    for track_id, data in track_log.items():
        duration = data["last_seen"] - data["first_seen"]
        row = (
            track_id,
            data["class_id"],
            data["class_name"],
            data["confidence"],
            data["first_seen"],
            data["last_seen"],
            duration
        )
        rows.append(row)
        cursor.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?)", row)

    conn.commit()
    conn.close()
    return rows


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    annotated_path = tempfile.mktemp(suffix="_annotated.mp4")
    track_log, frame_count, _ = runOnVideo(temp_path, output_path=annotated_path)

    rows = save_to_db(track_log, frame_count)

    return {
        "frames_processed": frame_count,
        "annotated_video": annotated_path,
        "detections": [
            {
                "track_id": r[0],
                "class_id": r[1],
                "class_name": r[2],
                "confidence": r[3],
                "first_seen": r[4],
                "last_seen": r[5],
                "duration": r[6]
            } for r in rows
        ]
    }


@app.post("/stream")
async def stream(file: UploadFile = File(...)):
    """Stream annotated frames as MJPEG. Saves results to DB and caches them for /results."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    session_id = str(uuid.uuid4())
    annotated_path = tempfile.mktemp(suffix="_annotated.mp4")

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    final_track_log = {}
    final_frame_count = 0

    def generate():
        nonlocal final_track_log, final_frame_count
        for frame_idx, (frame, track_log, _) in enumerate(runOnVideoStream(temp_path)):
            writer.write(frame)
            final_track_log = track_log
            final_frame_count = frame_idx + 1
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

        writer.release()

        # Save to DB and cache results once streaming is complete
        rows = save_to_db(final_track_log, final_frame_count)
        stream_results[session_id] = {
            "frames_processed": final_frame_count,
            "annotated_video": annotated_path,
            "detections": [
                {
                    "track_id": r[0],
                    "class_id": r[1],
                    "class_name": r[2],
                    "confidence": r[3],
                    "first_seen": r[4],
                    "last_seen": r[5],
                    "duration": r[6]
                } for r in rows
            ]
        }

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Session-ID": session_id}
    )


@app.get("/results/{session_id}")
def results(session_id: str):
    """Return cached results from a completed /stream session."""
    if session_id not in stream_results:
        return {"error": "Session not found or still in progress"}
    return stream_results.pop(session_id)  # pop to free memory after retrieval


@app.get("/download")
def download(path: str):
    """Serve an annotated video file back to the local machine."""
    return FileResponse(path, media_type="video/mp4", filename="annotated_output.mp4")