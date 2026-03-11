import tkinter as tk
from tkinter import filedialog
import requests

API_URL = "http://127.0.0.1:8000/infer"

def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not file_path:
        return

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "video/mp4")}
        response = requests.post(API_URL, files=files)
        data = response.json()
        show_results(data)

def show_results(data):
    win = tk.Toplevel(root)
    win.title("Inference Results")
    win.geometry("600x400")

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

# if __name__ == "__main__":
#     print("Starting app...")
#     upload_video()

# import demo as demo
# import tkinter as tk
# from tkinter import filedialog
# import requests
# from fastapi import UploadFile, File

# url = "http://127.0.0.1:8000/infer"

# files = {"file": open("frame.jpg", "rb")}

# response = requests.post(url, files=files)



# def upload_video():
#     file_path = filedialog.askopenfilename(
#         filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
#     )

#     if file_path:
#         result = demo.runOneFrame(file_path)
#         print(result)

# root = tk.Tk()
# root.title("Intersection Car Analyzer")
# root.geometry("400x200")

# upload_button = tk.Button(
#     root,
#     text="Upload Video",
#     command=upload_video,
#     height=2,
#     width=20
# )

# upload_button.pack(pady=50)

# root.mainloop()
# if __name__ == "__main__":
#     print("starting app")
#     upload_video()
