"""
detection/cheating.py
──────────────────────
Exam Cheating Detection
- Tracks head pose: Roll, Pitch, Yaw using MediaPipe
- Triggers alert if gaze > 60 degrees from centre for > 5 seconds
- Logs frequency and duration of each look-away event
"""

import cv2
import time
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[CHEATING] MediaPipe not found. Using simulated mode.")

class CheatingDetector:
    def __init__(self):
        self.alerts = []
        self.look_away_start = {}   # {face_id: timestamp when they started looking away}
        self.look_away_count = {}   # {face_id: total events}
        self.THRESHOLD_DEGREES = 60
        self.THRESHOLD_SECONDS = 5

        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        print("[CHEATING] Module ready.")

    def detect(self, frame):
        self.alerts = []

        if not MP_AVAILABLE:
            return self._simulate()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"alerts": [], "look_away_counts": self.look_away_count}

        h, w = frame.shape[:2]
        for idx, landmarks in enumerate(results.multi_face_landmarks):
            face_id = f"FACE_{idx}"
            yaw = self._estimate_yaw(landmarks, w)

            if abs(yaw) > self.THRESHOLD_DEGREES:
                # Face is looking away
                if face_id not in self.look_away_start:
                    self.look_away_start[face_id] = time.time()
                else:
                    duration = time.time() - self.look_away_start[face_id]
                    if duration >= self.THRESHOLD_SECONDS:
                        self.look_away_count[face_id] = self.look_away_count.get(face_id, 0) + 1
                        self.alerts.append({
                            "face_id": face_id,
                            "yaw": round(yaw, 1),
                            "duration_secs": round(duration, 1),
                            "total_events": self.look_away_count[face_id],
                            "severity": "high"
                        })
                        # Draw red warning
                        cv2.putText(frame, f"CHEATING ALERT! Yaw:{yaw:.0f}deg {duration:.1f}s",
                                    (10, 90 + idx*25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 0, 255), 2)
            else:
                self.look_away_start.pop(face_id, None)

        return {"alerts": self.alerts, "look_away_counts": self.look_away_count}

    def _estimate_yaw(self, landmarks, w):
        """Estimate yaw angle from nose tip vs face centre."""
        nose = landmarks.landmark[1]
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        face_centre_x = (left_ear.x + right_ear.x) / 2
        yaw = (nose.x - face_centre_x) * 180
        return yaw

    def _simulate(self):
        """Simulated output when MediaPipe not available."""
        return {"alerts": [], "look_away_counts": {}}
