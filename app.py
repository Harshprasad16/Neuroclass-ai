"""
Neuroclass.AI — Main Flask Application
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import cv2
import threading
import time
from datetime import datetime

from database.db import init_db, get_db
from detection.attendance import AttendanceDetector
from detection.cheating import CheatingDetector
from detection.drowsiness import DrowsinessDetector
from detection.phone import PhoneDetector
from detection.emotion import EmotionDetector
from detection.attention import AttentionDetector
from detection.intruder import IntruderDetector

app = Flask(__name__)
CORS(app)

# ── Global State ──────────────────────────────────────────────────────────────
latest_frame = None
detection_active = False
current_class_id = "CLASS_301"
camera_lock = threading.Lock()
results_lock = threading.Lock()

detection_results = {
    "attendance": {"present": 0, "enrolled": 0, "ratio": 0},
    "cheating_alerts": [],
    "drowsy_students": [],
    "phone_alerts": [],
    "emotions": {},
    "attention_scores": {"avg_score": 0},
    "intruders": [],
    "teacher_efficiency": 0,
    "total_present": 0,
    "total_enrolled": 0,
    "timestamp": "",
    "active": False
}

detectors = {}

# ── Init Detectors ────────────────────────────────────────────────────────────
def initialize_detectors(class_id):
    global detectors
    print("[NEUROCLASS] Loading detection modules...")
    detectors = {
        "attendance": AttendanceDetector(class_id=class_id),
        "cheating":   CheatingDetector(),
        "drowsiness": DrowsinessDetector(),
        "phone":      PhoneDetector(),
        "emotion":    EmotionDetector(),
        "attention":  AttentionDetector(),
        "intruder":   IntruderDetector(class_id=class_id),
    }
    print("[NEUROCLASS] All 7 modules ready.")

# ── Detection Loop ────────────────────────────────────────────────────────────
def detection_loop():
    global latest_frame, detection_results, detection_active
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check webcam connection.")
        detection_active = False
        return

    print("[NEUROCLASS] Camera open. Running at ~30 FPS...")
    frame_count = 0

    while detection_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        results = {}

        try:
            results["attendance"]  = detectors["attendance"].detect(frame)
            results["cheating"]    = detectors["cheating"].detect(frame)
            results["drowsiness"]  = detectors["drowsiness"].detect(frame)
            results["phone"]       = detectors["phone"].detect(frame)
            results["emotion"]     = detectors["emotion"].detect(frame)
            results["attention"]   = detectors["attention"].detect(frame)
            results["intruder"]    = detectors["intruder"].detect(frame)
        except Exception as e:
            print(f"[DETECTION ERROR] {e}")

        # Teacher Efficiency Score: (W1*E + W2*(1-D) + W3*A) * 100
        engagement  = results.get("attention", {}).get("avg_score", 0.5)
        present     = max(results.get("attendance", {}).get("present", 1), 1)
        distraction = len(results.get("phone", {}).get("alerts", [])) / present
        attend_r    = results.get("attendance", {}).get("ratio", 0.8)
        teacher_eff = round((0.5 * engagement + 0.3 * (1 - min(distraction, 1)) + 0.2 * attend_r) * 100, 1)

        with results_lock:
            detection_results.update({
                "attendance":        results.get("attendance", {}),
                "cheating_alerts":   results.get("cheating", {}).get("alerts", []),
                "drowsy_students":   results.get("drowsiness", {}).get("alerts", []),
                "phone_alerts":      results.get("phone", {}).get("alerts", []),
                "emotions":          results.get("emotion", {}),
                "attention_scores":  results.get("attention", {}),
                "intruders":         results.get("intruder", {}).get("intruders", []),
                "teacher_efficiency": teacher_eff,
                "total_present":     results.get("attendance", {}).get("present", 0),
                "total_enrolled":    results.get("attendance", {}).get("enrolled", 0),
                "timestamp":         datetime.now().strftime("%H:%M:%S"),
                "active":            True,
            })

        # Draw HUD on frame
        annotated = draw_hud(frame.copy(), teacher_eff)
        with camera_lock:
            latest_frame = annotated

        # Log to DB every 30 frames
        if frame_count % 30 == 0:
            log_to_db(results, teacher_eff)

        time.sleep(0.033)

    cap.release()
    with results_lock:
        detection_results["active"] = False
    print("[NEUROCLASS] Camera released.")


def draw_hud(frame, teacher_eff):
    """Draw HUD overlay on video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (4, 8, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "NEUROCLASS.AI  LIVE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
    cv2.putText(frame, f"Efficiency: {teacher_eff}%  |  {datetime.now().strftime('%H:%M:%S')}",
                (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 150), 1)
    return frame


def log_to_db(results, teacher_eff):
    try:
        db = get_db(current_class_id)
        db.execute("""
            INSERT INTO session_logs
            (timestamp, present_count, teacher_efficiency, phone_alerts, drowsy_count, intruder_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            results.get("attendance", {}).get("present", 0),
            teacher_eff,
            len(results.get("phone", {}).get("alerts", [])),
            len(results.get("drowsiness", {}).get("alerts", [])),
            len(results.get("intruder", {}).get("intruders", [])),
        ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB LOG ERROR] {e}")


def generate_frames():
    while True:
        with camera_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    global detection_active, current_class_id
    data = request.get_json() or {}
    current_class_id = data.get("class_id", current_class_id)
    if not detection_active:
        detection_active = True
        initialize_detectors(current_class_id)
        t = threading.Thread(target=detection_loop, daemon=True)
        t.start()
        return jsonify({"status": "started", "class_id": current_class_id})
    return jsonify({"status": "already_running", "class_id": current_class_id})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"status": "stopped"})

@app.route('/api/status')
def get_status():
    with results_lock:
        return jsonify(detection_results)

@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance')
def get_attendance():
    db = get_db(current_class_id)
    rows = db.execute(
        "SELECT * FROM attendance WHERE date = date('now') ORDER BY time DESC"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/events')
def get_events():
    db = get_db(current_class_id)
    rows = db.execute(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/register_student', methods=['POST'])
def register_student():
    data = request.get_json()
    student_id = data.get("student_id")
    name = data.get("name")
    db = get_db(current_class_id)
    db.execute(
        "INSERT OR REPLACE INTO students (student_id, name, class_id) VALUES (?, ?, ?)",
        (student_id, name, current_class_id)
    )
    db.commit()
    db.close()
    return jsonify({"status": "registered", "student_id": student_id, "name": name})

@app.route('/api/session_history')
def session_history():
    db = get_db(current_class_id)
    rows = db.execute(
        "SELECT * FROM session_logs ORDER BY timestamp DESC LIMIT 30"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/log_alert', methods=['POST'])
def log_alert():
    data = request.get_json()
    db = get_db(current_class_id)
    db.execute(
        "INSERT INTO events (timestamp, event_type, severity, seat, description) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(),
         data.get("type", "unknown"),
         data.get("severity", "medium"),
         data.get("seat", "N/A"),
         data.get("description", ""))
    )
    db.commit()
    db.close()
    return jsonify({"status": "logged"})

@app.route('/api/classes')
def list_classes():
    import os
    dbs = [f.replace(".db","") for f in os.listdir("database/") if f.endswith(".db")]
    return jsonify({"classes": dbs})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("   NEUROCLASS.AI — Classroom Intelligence System")
    print("="*55)
    init_db(current_class_id)
    print(f"[DB] Class database ready: {current_class_id}")
    print(f"[SERVER] Running at http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
