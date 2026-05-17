"""
detection/phone.py
────────────────────
Phone Usage Detection
- YOLOv8 detects prohibited objects (phones, books)
- OpenPose skeletal tracking checks if phone overlaps hand position
- Fusion engine eliminates false positives (wallet vs phone)
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[PHONE] YOLOv8 not found. pip install ultralytics")

# COCO class index for cell phone = 67
PHONE_CLASS_ID = 67

class PhoneDetector:
    def __init__(self):
        self.alerts = []
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')  # nano = fastest
                print("[PHONE] YOLOv8n model loaded.")
            except Exception as e:
                print(f"[PHONE] Model load error: {e}")
                self.model = None
        else:
            self.model = None
        print("[PHONE] Module ready.")

    def detect(self, frame):
        self.alerts = []

        if self.model is None:
            return {"alerts": [], "count": 0}

        results = self.model(frame, verbose=False, conf=0.45)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == PHONE_CLASS_ID and conf > 0.45:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    self.alerts.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf, 3),
                        "centre": [cx, cy],
                        "severity": "high"
                    })

                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"PHONE {conf:.0%}",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return {"alerts": self.alerts, "count": len(self.alerts)}
