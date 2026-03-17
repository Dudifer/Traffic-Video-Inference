import tkinter as tk
from tkinter import filedialog
import requests
import subprocess
import sys

API_URL = "http://127.0.0.1:8000/infer"

def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not file_path:
        return

    root.withdraw()  # Hide the main window while processing

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "video/mp4")}
            response = requests.post(API_URL, files=files)
            data = response.json()
            show_results(data)
    except Exception as e:
        root.deiconify()  # Restore window on error
        raise e

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

    # Restore main window when results window is closed
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

    # Show annotated video path and open button if present
    annotated_path = data.get("annotated_video")
    if annotated_path:
        frame = tk.Frame(win, pady=6)
        frame.pack(fill=tk.X, padx=10)
        tk.Label(frame, text=f"Annotated video: {annotated_path}", anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(frame, text="▶ Open", command=lambda: open_video(annotated_path)).pack(side=tk.RIGHT)

root = tk.Tk()
root.title("Intersection Car Analyzer")
root.geometry("800x400")

upload_button = tk.Button(
    root,
    text="Upload Video",
    command=upload_video,
    height=4,
    width=24
)

upload_button.pack(pady=50)

root.mainloop()