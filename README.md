# Neuroclass.AI — Backend Setup Guide (Phase 1)
## Complete step-by-step for Windows PC + Webcam

---

## WHAT YOU NEED
- Windows PC with a webcam
- Python 3.10 installed (https://python.org/downloads)
- Internet connection (for downloading models)

---

## STEP 1 — Install Python 3.10

1. Go to https://python.org/downloads
2. Download **Python 3.10.x** (NOT 3.12 — mediapipe needs 3.10)
3. During install: CHECK the box "Add Python to PATH"
4. Click Install Now

Verify: open CMD and type:
```
python --version
```
Should show: Python 3.10.x

---

## STEP 2 — Download the Project Files

Put all these files in ONE folder on your PC, for example:
```
C:\Users\harsh\Desktop\neuroclass\
    app.py
    requirements.txt
    database\
        __init__.py
        db.py
    detection\
        __init__.py
        attendance.py
        cheating.py
        drowsiness.py
        phone.py
        emotion.py
        attention.py
        intruder.py
    templates\
        dashboard.html
```

---

## STEP 3 — Open CMD in the Project Folder

1. Open File Explorer
2. Go to your neuroclass folder
3. Click the address bar → type `cmd` → press Enter

---

## STEP 4 — Create a Virtual Environment

Type these commands one by one:

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command line.

---

## STEP 5 — Install All Libraries

```cmd
pip install flask flask-cors opencv-python numpy
pip install mediapipe
pip install ultralytics
pip install deepface
```

> NOTE: These downloads may take 5-10 minutes (models are large).
> If deepface fails, skip it — emotion detection will run in simulation mode.

---

## STEP 6 — Run the Server

```cmd
python app.py
```

You should see:
```
===================================================
   NEUROCLASS.AI — Classroom Intelligence System
===================================================
[DB] Tables ready for CLASS_301
[SERVER] Running at http://localhost:5000
```

---

## STEP 7 — Open the Dashboard

Open your browser and go to:
```
http://localhost:5000
```

You will see the Neuroclass.AI live dashboard!

---

## STEP 8 — Start Detection

1. Select a classroom (Room 301, 302, etc.)
2. Click **▶ Start Detection**
3. Allow camera access if prompted
4. You will see:
   - Live webcam feed with AI overlays
   - Real-time engagement heatmap
   - Teacher Efficiency Score updating
   - Live event log

---

## API ENDPOINTS (for testing)

You can test these in your browser or Postman:

| Endpoint | Method | What it does |
|---|---|---|
| / | GET | Live dashboard |
| /api/start | POST | Start detection |
| /api/stop | POST | Stop detection |
| /api/status | GET | Get live detection results |
| /api/video_feed | GET | MJPEG live video stream |
| /api/attendance | GET | Today's attendance |
| /api/events | GET | Last 50 events |
| /api/session_history | GET | Session logs |
| /api/register_student | POST | Register a new student |

### Example: Register a student
```
POST http://localhost:5000/api/register_student
Content-Type: application/json

{
  "student_id": "STU001",
  "name": "Harsh Prasad"
}
```

### Example: Set class
```
POST http://localhost:5000/api/set_class
Content-Type: application/json

{
  "class_id": "CLASS_302"
}
```

---

## DATABASE LOCATION

Each class gets its own database file:
```
database/
    CLASS_301.db    ← Room 301 data
    CLASS_302.db    ← Room 302 data
    CLASS_303.db    ← Room 303 data
```

Data never mixes between classes.

---

## DETECTION MODULES STATUS

| Module | Status | Notes |
|---|---|---|
| Attendance | ✅ Running | Uses OpenCV Haar Cascade |
| Cheating | ✅ Running | Needs MediaPipe |
| Drowsiness | ✅ Running | Needs MediaPipe |
| Phone Usage | ✅ Running | Needs YOLOv8 (auto-downloads) |
| Emotion | ⚠ Simulated | Needs DeepFace (optional) |
| Attention | ✅ Running | Needs MediaPipe |
| Intruder | ✅ Running | Full matching needs face enrollment |

---

## COMMON ERRORS & FIXES

**"Camera not found"**
→ Check webcam is connected. Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in app.py if you have multiple cameras.

**"ModuleNotFoundError: mediapipe"**
→ Run: `pip install mediapipe`

**"ModuleNotFoundError: ultralytics"**
→ Run: `pip install ultralytics`

**Port 5000 already in use**
→ Change `port=5000` to `port=5001` in app.py and visit http://localhost:5001

**Detection is slow**
→ Normal on CPU. For faster processing, reduce frame resolution in app.py:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## PHASE 2 (Coming Next)
- Deploy to Render/Heroku (free cloud hosting)
- Switch to PostgreSQL
- Teacher gets dashboard from anywhere
- Mobile notifications

## PHASE 3 (Coming Next)
- Deploy detection to Raspberry Pi / NVIDIA Jetson
- Pi streams data to cloud API
- Full production classroom system

---

Built by Harsh Prasad | Asansol Engineering College
GitHub: https://github.com/Harshprasad16/Neuroclass-ai
