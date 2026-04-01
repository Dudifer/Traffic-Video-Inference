# Traffic Video Inference

> A program for getting real-time vehicle detection made to simulate roadside hardware getting cloud-based ML inference. 

---

## Demo
<iframe width="560" height="315" src="https://drive.google.com/file/d/1bsneTHbuaZo2mqn0M4hQeYlQUit8x3CN/view?usp=drive_link" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Architecture

![alt text](<Screenshot 2026-03-25 180207.png>)

| Component | Where | Responsibility |
|---|---|---|
| `app.py` | Local | Tkinter GUI — file selection, live preview toggle, results display, video download |
| `client.py` | Server | FastAPI server — `/infer`, `/stream`, `/download` endpoints |
| `model.py` | Server | YOLOv8 + ByteTrack — full-video tracking, frame annotation, output writing |
| `detections.db` | Server | SQLite — persists track ID, class, confidence, first/last seen, duration |
| EC2 instance | Cloud | Hosts FastAPI + model, port 8000 open to internet |

---

## Screenshots

### Upload window
![alt text](<Screenshot 2026-03-25 175821.png>)

### Live preview
![alt text](<Screenshot 2026-03-25 175858.png>)

### Inference results
![alt text](<Screenshot 2026-03-25 175943.png>)

---

## How to run

### Server (EC2)

SSH into the instance:
```bash
ssh -i your-key.pem ubuntu@<EC2-IP>
```

Activate the virtual environment and start the server:
```bash
source venv/bin/activate
python -m uvicorn client:app --host 0.0.0.0 --port 8000
```

Make sure port 8000 is open in your EC2 security group (Custom TCP, source `0.0.0.0/0`).

### Local client

Install dependencies:
```bash
pip install requests opencv-python
```

Update the server IP in `app.py`:
```python
SERVER = "http://<EC2-IP>:8000"
```

Run the app:
```bash
python app.py
```

---

## Design decisions & tradeoffs

**Why cloud inference?**

Running inference in the cloud means model updates don't require redeploying edge hardware, compute can be scaled centrally, and more powerful instances can be used without changing the client.

| Decision | Advantage | Tradeoff |
|---|---|---|
| Cloud inference | Model updates without redeploying edge hardware; centralized scaling | Network latency per request; requires stable internet |
| MJPEG streaming | Real-time visual feedback before inference completes | Higher bandwidth usage; frame drops on slow connections |
| Full-video `model.track()` | Tracker state persists correctly across all frames; no per-frame model reloads | Full video must upload before inference begins |
| SQLite persistence | Zero-config, no external DB, fully portable | Not suitable for concurrent multi-user or large-scale deployments |
| Tkinter GUI | Ships with Python, no extra install on the client machine | Limited styling; not suitable for web or mobile distribution |

---

## Performance

| Metric | Value |
|---|---|
| Average inference time | 50 ms/frame |
| EC2 instance type | m7i-flex.large |
| Video resolution tested | 720p |
| Model | YOLOv8n (nano) |

---

## Future improvements

- Async job queue (e.g. Celery + Redis) to handle concurrent video uploads without blocking
- Batch inference to process multiple frames simultaneously for higher throughput
- Persistent EC2 IP via Elastic IP, or a domain name via Route 53
- Edge hardware optimization — quantized models (YOLOv8n-int8) for Raspberry Pi or Jetson Nano
- Streamlit dashboard for browsing and filtering historical detections from the SQLite database
- Upgrade to a GPU-enabled instance (e.g. g4dn.xlarge) for significantly faster inference — the current m7i-flex.large is CPU-only, which bottlenecks throughput on longer or higher-resolution videos