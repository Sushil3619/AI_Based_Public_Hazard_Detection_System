"""
============================================================
  YOLO Pose: Crowd, Fall & SOS Emergency Detection
  - Pose Estimation (yolov8n-pose.pt)
  - Fallen Person Detection
  - SOS / 'X' Sign Help Detection (Hands Up & Crossed)
  - Independent Alarms for all 3 events
============================================================
"""

import cv2
import pygame 
import os     
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
CROWD_THRESHOLD   = 3     
FALLEN_RATIO      = 1.0   
CONFIDENCE_MIN    = 0.5   # Keypoint confidence

# तीन स्वतंत्र ऑडिओ फाईल्स
CROWD_SOUND_FILE  = "alarm2.mp3" 
FALLEN_SOUND_FILE = "alarm.wav"  
HELP_SOUND_FILE   = "emergency_unit.mp3"

# ── Colors (BGR format for OpenCV) ──────────────────────────
COLOR_NORMAL  = (0, 220, 0)     
COLOR_FALLEN  = (0, 0, 255)     
COLOR_SOS     = (255, 0, 255)   # जांभळा (Purple) -> SOS साठी
COLOR_ALERT   = (0, 140, 255)   
COLOR_INFO    = (255, 255, 255) 

# ────────────────────────────────────────────────────────────

def draw_banner(frame, text, color, y_start=0):
    """Screen वर आडवा banner दाखवतो."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + 50), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, y_start + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

def analyze_pose_and_boxes(results):
    """
    माणूस पडलाय का आणि SOS (Help) मागतोय का हे तपासतो.
    """
    person_data = []
    fallen_count = 0
    sos_count = 0

    boxes = results[0].boxes
    keypoints = results[0].keypoints

    if boxes is None or keypoints is None:
        return 0, 0, 0, []

    for i in range(len(boxes)):
        # 1. Bounding Box Logic (Fallen Detection)
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        bw, bh = (x2 - x1), (y2 - y1)
        ratio = bw / bh if bh > 0 else 0
        is_fallen = ratio > FALLEN_RATIO
        
        if is_fallen:
            fallen_count += 1

        # 2. Pose Logic (SOS / 'X' Sign Detection)
        is_sos = False
        kpts = keypoints.data[i] # Shape (17, 3) -> x, y, confidence
        
        # YOLOv8 Pose Indices: 5:L-Shoulder, 6:R-Shoulder, 9:L-Wrist, 10:R-Wrist
        ls_x, ls_y, ls_c = kpts[5]
        rs_x, rs_y, rs_c = kpts[6]
        lw_x, lw_y, lw_c = kpts[9]
        rw_x, rw_y, rw_c = kpts[10]

        # सर्व आवश्यक keypoints व्यवस्थित दिसत असतील तरच चेक करा
        if all(c > CONFIDENCE_MIN for c in [ls_c, rs_c, lw_c, rw_c]):
            
            # नियम १: दोन्ही मनगटे खांद्याच्या वर आहेत का? (Y-coordinate कमी असणे)
            hands_up = (lw_y < ls_y) and (rw_y < rs_y)
            
            # नियम २: 'X' Sign (मनगटांमधील अंतर खांद्याच्या रुंदीपेक्षा कमी/क्रॉस आहे का?)
            shoulder_width = abs(ls_x - rs_x)
            wrist_dist = abs(lw_x - rw_x)
            
            # जर हात वर असतील आणि मनगटे एकमेकांच्या खूप जवळ असतील (किंवा क्रॉस)
            if hands_up and wrist_dist < (shoulder_width * 0.8): 
                is_sos = True
                sos_count += 1

        person_data.append({
            "bbox": (x1, y1, x2, y2),
            "is_fallen": is_fallen,
            "is_sos": is_sos
        })

    return len(boxes), fallen_count, sos_count, person_data

def draw_detections(frame, person_data):
    """Boxes आणि Labels frame वर काढतो."""
    for r in person_data:
        x1, y1, x2, y2 = r["bbox"]
        
        if r["is_sos"]:
            color = COLOR_SOS
            label = "HELP NEEDED (SOS)!"
        elif r["is_fallen"]:
            color = COLOR_FALLEN
            label = "FALLEN PERSON!"
        else:
            color = COLOR_NORMAL
            label = "Person"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def overlay_stats(frame, total, fallen, sos, is_crowd):
    """Stats आणि स्वतंत्र बॅनर्स दाखवतो."""
    h, w = frame.shape[:2]

    info_lines = [
        f"Persons Detected : {total}",
        f"Fallen Persons   : {fallen}",
        f"SOS Requests     : {sos}"
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 80 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)

    # UI Banners (Highest priority on top)
    y_offset = h - 60
    
    if not is_crowd and fallen == 0 and sos == 0:
        draw_banner(frame, "Status: Normal", (40, 160, 40), y_start=y_offset)
        return

    if is_crowd:
        draw_banner(frame, "⚠ CROWD LIMIT EXCEEDED!", COLOR_ALERT, y_start=y_offset)
        y_offset -= 55
        
    if fallen > 0:
        draw_banner(frame, "⚠ EMERGENCY: FALLEN PERSON!", COLOR_FALLEN, y_start=y_offset)
        y_offset -= 55
        
    if sos > 0:
        draw_banner(frame, "🚨 SOS: SOMEONE NEEDS HELP!", COLOR_SOS, y_start=y_offset)

def main():
    print("=" * 60)
    print("  Smart Rescue System: Pose & Anomaly Detection")
    print("=" * 60)

    # ── Pygame Audio Setup ───────────────────────────────
    pygame.mixer.init()
    
    # 3 स्वतंत्र चॅनेल्स 
    ch_crowd = pygame.mixer.Channel(0)
    ch_fall  = pygame.mixer.Channel(1)
    ch_sos   = pygame.mixer.Channel(2)

    snd_crowd, snd_fall, snd_sos = None, None, None

    if os.path.exists(CROWD_SOUND_FILE): snd_crowd = pygame.mixer.Sound(CROWD_SOUND_FILE)
    if os.path.exists(FALLEN_SOUND_FILE): snd_fall = pygame.mixer.Sound(FALLEN_SOUND_FILE)
    if os.path.exists(HELP_SOUND_FILE): snd_sos = pygame.mixer.Sound(HELP_SOUND_FILE)
    
    # ── YOLO Pose Model ──────────────────────────────────
    # टीप: पहिल्यांदा रन करताना हे मॉडेल इंटरनेटवरून डाउनलोड होईल.
    model = YOLO("yolov8n-pose.pt")
    print("[✓] YOLO Pose model लोड झाले!\n")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        
        total, fallen, sos, person_data = analyze_pose_and_boxes(results)
        is_crowd = (total >= CROWD_THRESHOLD)

        # ── Audio Playback Logic ──────────────────────────────
        # 1. Crowd Logic
        if is_crowd and snd_crowd:
            if not ch_crowd.get_busy(): ch_crowd.play(snd_crowd, loops=-1)
        elif not is_crowd and ch_crowd.get_busy(): ch_crowd.stop()

        # 2. Fall Logic
        if fallen > 0 and snd_fall:
            if not ch_fall.get_busy(): ch_fall.play(snd_fall, loops=-1)
        elif fallen == 0 and ch_fall.get_busy(): ch_fall.stop()

        # 3. SOS (Help) Logic
        if sos > 0 and snd_sos:
            if not ch_sos.get_busy(): ch_sos.play(snd_sos, loops=-1)
        elif sos == 0 and ch_sos.get_busy(): ch_sos.stop()

        # ── UI Visualization ─────────────────────────────────
        draw_banner(frame, "Smart Rescue Camera  |  Press 'q' to Quit", (30, 30, 30))
        draw_detections(frame, person_data)
        overlay_stats(frame, total, fallen, sos, is_crowd)

        cv2.imshow("Smart Rescue System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()