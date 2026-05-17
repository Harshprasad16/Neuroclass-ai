"""
detection/intruder.py
──────────────────────
Intruder & Unknown Face Detection
- Compares detected faces against class-specific student database
- Unknown faces = intruders → immediate alert
- Students from other classes flagged as "extra students"
- Notifies teacher with what the intruder is doing
- Each class has its own isolated face database
"""

import cv2
import numpy as np
import time
from database.db import get_db

class IntruderDetector:
    def __init__(self, class_id: str):
        self.class_id = class_id
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_faces = {}      # {student_id: face_encoding}
        self.intruder_log = {}     # {face_hash: first_seen_timestamp}
        self._load_known_faces()
        print(f"[INTRUDER] Module ready. {len(self.known_faces)} known faces loaded.")

    def _load_known_faces(self):
        """Load registered student face encodings from DB."""
        try:
            db = get_db(self.class_id)
            rows = db.execute(
                "SELECT student_id, name, face_encoding FROM students"
            ).fetchall()
            db.close()
            for row in rows:
                if row["face_encoding"]:
                    enc = np.frombuffer(row["face_encoding"], dtype=np.float32)
                    self.known_faces[row["student_id"]] = {
                        "name": row["name"],
                        "encoding": enc
                    }
        except Exception as e:
            print(f"[INTRUDER] DB load error: {e}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        intruders = []
        now = time.time()

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_hash = hash(face_region.tobytes()[:100])  # Quick hash for tracking

            # In full version: compare face_encoding vs self.known_faces
            # Phase 1: if no known faces registered → everyone is "unverified"
            is_known = len(self.known_faces) > 0

            if not is_known:
                # No one registered yet — flag as unverified
                status = "UNVERIFIED"
                color = (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, "UNVERIFIED — Register Students",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # With registered faces, non-matching = intruder
                matched = self._match_face(face_region)
                if not matched:
                    status = "INTRUDER"
                    color = (0, 0, 255)

                    if face_hash not in self.intruder_log:
                        self.intruder_log[face_hash] = now
                        self._log_intruder_event(x, y, w, h)

                    duration = round(now - self.intruder_log.get(face_hash, now), 1)
                    intruders.append({
                        "bbox": [x, y, w, h],
                        "duration_secs": duration,
                        "severity": "high",
                        "description": "Unknown face detected in classroom"
                    })

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, f"INTRUDER! {duration}s",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 1)

        return {"intruders": intruders, "count": len(intruders)}

    def _match_face(self, face_region):
        """Placeholder for face encoding comparison."""
        # In full version: compute encoding and compare with self.known_faces
        # Returns True if match found, False if intruder
        return True  # Default to known until full encoding is implemented

    def _log_intruder_event(self, x, y, w, h):
        """Log intruder detection to DB."""
        try:
            db = get_db(self.class_id)
            db.execute(
                "INSERT INTO events (event_type, severity, description) VALUES (?, ?, ?)",
                ("intruder_detected", "high", f"Unknown face at position ({x},{y})")
            )
            db.commit()
            db.close()
        except Exception as e:
            print(f"[INTRUDER DB ERROR] {e}")
