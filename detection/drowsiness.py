"""
detection/drowsiness.py
────────────────────────
Drowsiness Detection using EAR + PERCLOS
- Eye Aspect Ratio (EAR): distance between eye landmarks
- PERCLOS: percentage of eye closure over a time window
- Triggers alert when eyes closed beyond threshold
"""

import cv2
import time
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# EAR threshold — below this = eyes closing
EAR_THRESHOLD = 0.22
# PERCLOS threshold — if eyes closed >30% of last 60 frames = drowsy
PERCLOS_THRESHOLD = 0.30
WINDOW_SIZE = 60  # frames

class DrowsinessDetector:
    def __init__(self):
        # LEFT eye landmarks (MediaPipe)
        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33,  160, 158, 133, 153, 144]
        self.ear_history = {}  # {face_id: [ear values]}
        self.alerts = []

        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=10, min_detection_confidence=0.5
            )
        print("[DROWSINESS] Module ready.")

    def _ear(self, landmarks, eye_points, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C) if C > 0 else 0

    def detect(self, frame):
        self.alerts = []
        if not MP_AVAILABLE:
            return {"alerts": []}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"alerts": []}

        for idx, lm in enumerate(results.multi_face_landmarks):
            face_id = f"FACE_{idx}"
            lms = lm.landmark

            left_ear  = self._ear(lms, self.LEFT_EYE, w, h)
            right_ear = self._ear(lms, self.RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Update history
            if face_id not in self.ear_history:
                self.ear_history[face_id] = []
            self.ear_history[face_id].append(ear)
            if len(self.ear_history[face_id]) > WINDOW_SIZE:
                self.ear_history[face_id].pop(0)

            # PERCLOS — ratio of frames where eyes were closed
            history = self.ear_history[face_id]
            closed_frames = sum(1 for e in history if e < EAR_THRESHOLD)
            perclos = closed_frames / len(history) if history else 0

            status = "awake"
            color = (0, 255, 100)

            if perclos > PERCLOS_THRESHOLD:
                status = "DROWSY"
                color = (0, 0, 255)
                self.alerts.append({
                    "face_id": face_id,
                    "ear": round(ear, 3),
                    "perclos": round(perclos, 3),
                    "severity": "high"
                })

            # Draw EAR on frame
            nose_y = int(lms[1].y * h)
            cv2.putText(frame, f"EAR:{ear:.2f} PERCLOS:{perclos:.0%} [{status}]",
                        (10, 120 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return {"alerts": self.alerts, "count": len(self.alerts)}
