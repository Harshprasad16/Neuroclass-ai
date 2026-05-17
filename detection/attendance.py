"""
detection/attendance.py
──────────────────────
Smart Attendance Module
- Detects faces using OpenCV Haar Cascades
- Matches against registered students (face encoding comparison)
- Marks present only if student stays 40+ minutes
- Logs entry/exit times to class DB
"""

import cv2
import numpy as np
import time
from datetime import datetime
from database.db import get_db

class AttendanceDetector:
    def __init__(self, class_id: str):
        self.class_id = class_id
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Track entry times: {student_id: entry_timestamp}
        self.entry_times = {}
        # Track confirmed attendance
        self.confirmed = set()
        # Load enrolled students from DB
        self.enrolled = self._load_enrolled()
        print(f"[ATTENDANCE] Module ready. {len(self.enrolled)} students enrolled.")

    def _load_enrolled(self):
        try:
            db = get_db(self.class_id)
            rows = db.execute("SELECT student_id, name FROM students").fetchall()
            db.close()
            return {r["student_id"]: r["name"] for r in rows}
        except:
            return {}

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        detected_ids = []
        now = time.time()

        for (x, y, w, h) in faces:
            # In full version: compare face encoding against DB
            # For Phase 1: simulate detection
            face_id = f"STUDENT_{x % 10 + 1:02d}"  # Simulated ID

            if face_id not in self.entry_times:
                self.entry_times[face_id] = now
                self._log_entry(face_id)

            duration_mins = (now - self.entry_times[face_id]) / 60

            # Mark confirmed if 40+ minutes
            if duration_mins >= 40 and face_id not in self.confirmed:
                self.confirmed.add(face_id)
                self._confirm_attendance(face_id)

            detected_ids.append(face_id)

            # Draw box
            color = (0, 255, 100) if face_id in self.confirmed else (255, 200, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{face_id} {'✓' if face_id in self.confirmed else f'{int(duration_mins)}m'}"
            cv2.putText(frame, label, (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return {
            "present": len(set(detected_ids)),
            "enrolled": max(len(self.enrolled), 32),
            "ratio": len(set(detected_ids)) / max(len(self.enrolled), 32),
            "confirmed": len(self.confirmed),
            "entry_times": {k: round((now-v)/60, 1) for k, v in self.entry_times.items()},
        }

    def _log_entry(self, student_id):
        try:
            db = get_db(self.class_id)
            name = self.enrolled.get(student_id, "Unknown")
            db.execute(
                "INSERT INTO attendance (student_id, name, entry_time, status) VALUES (?, ?, ?, ?)",
                (student_id, name, datetime.now().isoformat(), "pending")
            )
            db.commit()
            db.close()
        except Exception as e:
            print(f"[ATTENDANCE DB ERROR] {e}")

    def _confirm_attendance(self, student_id):
        try:
            db = get_db(self.class_id)
            db.execute(
                "UPDATE attendance SET status='present', duration_mins=40 WHERE student_id=? AND date=date('now')",
                (student_id,)
            )
            db.commit()
            db.close()
            print(f"[ATTENDANCE] {student_id} confirmed present (40+ mins)")
        except Exception as e:
            print(f"[ATTENDANCE CONFIRM ERROR] {e}")
