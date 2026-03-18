import tkinter as tk
from tkinter import filedialog
import requests
import subprocess
import sys
import threading
import os
import tempfile
import cv2
import numpy as np

SERVER = "http://3.148.247.151:8000"

def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not file_path:
        return

    root.withdraw()

    if show_var.get():
        # Send to /stream and display frames locally
        threading.Thread(target=stream_video, args=(file_path,), daemon=True).start()
    else:
        # Send to /infer and show results table
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f, "video/mp4")}
                response = requests.post(f"{SERVER}/infer", files=files)
                data = response.json()
                show_results(data)
        except Exception as e:
            root.deiconify()
            raise e


def stream_video(file_path: str):
    """POST video to /stream and display MJPEG frames in a local OpenCV window."""
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{SERVER}/stream",
            files={"file": (file_path, f, "video/mp4")},
            stream=True
        )

    buffer = b""
    for chunk in response.iter_content(chunk_size=4096):
        buffer += chunk
        # Each MJPEG frame is delimited by --frame boundaries
        start = buffer.find(b"\xff\xd8")  # JPEG start
        end = buffer.find(b"\xff\xd9")    # JPEG end
        if start != -1 and end != -1:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Live Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    root.after(0, root.deiconify)


def download_and_open(remote_path):
    """Download the annotated video from the server and open it locally."""
    response = requests.get(f"{SERVER}/download", params={"path": remote_path}, stream=True)
    local_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    open_video(local_path)

def open_video(path):
    """Open the annotated video in the system default player."""
    if sys.platform == "win32":
        subprocess.Popen(["start", "", path], shell=True)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def show_results(data):
    win = tk.Toplevel(root)
    win.title("Inference Results")
    win.geometry("600x450")

    win.protocol("WM_DELETE_WINDOW", lambda: [win.destroy(), root.deiconify()])

    text = tk.Text(win, wrap=tk.WORD, padx=10, pady=10)
    text.pack(expand=True, fill=tk.BOTH)

    text.insert(tk.END, f"Frames processed: {data['frames_processed']}\n\n")
    text.insert(tk.END, f"{'Track ID':<12}{'Class ID':<12}{'Confidence':<14}{'Duration (s)':<12}\n")
    text.insert(tk.END, "-" * 50 + "\n")

    for d in data["detections"]:
        text.insert(tk.END,
            f"{d['track_id']:<12}{d['class_id']:<12}{d['confidence']:<14.2f}{d['duration']:<12.2f}\n"
        )

    text.config(state=tk.DISABLED)

    annotated_path = data.get("annotated_video")
    if annotated_path:
        frame = tk.Frame(win, pady=6)
        frame.pack(fill=tk.X, padx=10)
        tk.Label(frame, text=f"Annotated video: {annotated_path}", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(frame, text="▶ Open", command=lambda: threading.Thread(target=download_and_open, args=(annotated_path,), daemon=True).start()).pack(side=tk.RIGHT)

root = tk.Tk()
root.title("Intersection Car Analyzer")
root.geometry("800x400")

show_var = tk.BooleanVar(value=False)

upload_button = tk.Button(
    root,
    text="Upload Video",
    command=upload_video,
    height=4,
    width=24
)
upload_button.pack(pady=50)

tk.Checkbutton(
    root,
    text="Show live preview during inference",
    variable=show_var
).pack()

root.mainloop()