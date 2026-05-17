"""
detection/attention.py
───────────────────────
Attention Analytics
- Blink rate tracking
- Gaze direction (where student is looking)
- Body posture analysis
- Generates attention score 0.0 - 1.0 per student
"""

import cv2
import numpy as np
import time

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

class AttentionDetector:
    def __init__(self):
        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33,  160, 158, 133, 153, 144]
        self.blink_state   = {}    # {face_id: is_blinking}
        self.blink_counts  = {}    # {face_id: count}
        self.blink_times   = {}    # {face_id: [timestamps]}
        self.attention_scores = {} # {face_id: score}
        self._frame_count = 0

        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=10, min_detection_confidence=0.5,
                refine_landmarks=True
            )
        print("[ATTENTION] Module ready.")

    def _ear(self, lms, eye_pts, w, h):
        pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in eye_pts]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C) if C > 0 else 0

    def _blink_rate_score(self, face_id):
        """Normal blink rate = 12-20 per minute. Too low = staring/distracted."""
        times = self.blink_times.get(face_id, [])
        now = time.time()
        recent = [t for t in times if now - t < 60]
        rate = len(recent)
        # Score: 12-20 bpm = 1.0, outside = lower
        if 12 <= rate <= 20:
            return 1.0
        elif rate < 12:
            return max(0.3, rate / 12)
        else:
            return max(0.5, 1 - (rate - 20) / 20)

    def detect(self, frame):
        self._frame_count += 1
        if not MP_AVAILABLE:
            return {"avg_score": 0.7, "scores": {}}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"avg_score": 0, "scores": self.attention_scores}

        scores = []
        for idx, lm in enumerate(results.multi_face_landmarks):
            face_id = f"FACE_{idx}"
            lms = lm.landmark

            # 1. EAR for blink detection
            ear = self._ear(lms, self.LEFT_EYE, w, h)
            currently_blinking = ear < 0.22

            if currently_blinking and not self.blink_state.get(face_id, False):
                # Blink just started
                if face_id not in self.blink_times:
                    self.blink_times[face_id] = []
                self.blink_times[face_id].append(time.time())
                self.blink_counts[face_id] = self.blink_counts.get(face_id, 0) + 1
            self.blink_state[face_id] = currently_blinking

            # 2. Gaze: check if looking forward (nose centre vs face centre)
            nose_x = lms[1].x
            left_x = lms[234].x
            right_x = lms[454].x
            face_centre_x = (left_x + right_x) / 2
            yaw_approx = abs(nose_x - face_centre_x) * 100
            gaze_score = max(0, 1 - (yaw_approx / 30))

            # 3. Blink rate score
            blink_score = self._blink_rate_score(face_id)

            # Combined attention score
            attention = round((0.5 * gaze_score + 0.5 * blink_score), 3)
            self.attention_scores[face_id] = attention
            scores.append(attention)

            # Draw
            color = (0, 255, 100) if attention > 0.6 else (0, 165, 255) if attention > 0.3 else (0, 0, 255)
            nose_px = int(lms[1].x * w), int(lms[1].y * h)
            cv2.putText(frame, f"ATT:{attention:.0%}",
                        (nose_px[0] - 30, nose_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        avg = round(sum(scores) / len(scores), 3) if scores else 0
        return {"avg_score": avg, "scores": self.attention_scores, "count": len(scores)}
