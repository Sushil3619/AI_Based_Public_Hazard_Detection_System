"""
============================================================
  Smart Rescue System: YOLO Pose + Hardware GPS (NEO-6M)
  - Pose Estimation & Anomaly Detection
  - Real-Time Hardware GPS Tracking (NEO-6M via Serial)
  - Threading used for Zero Video Lag
============================================================
"""

import cv2
import pygame 
import os     
import serial   # <-- नवीन: USB Port वाचण्यासाठी
import pynmea2  # <-- नवीन: GPS चा डेटा सोपा करण्यासाठी
import threading # <-- नवीन: कॅमेरा लॅग होऊ नये म्हणून
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
CROWD_THRESHOLD   = 5    
FALLEN_RATIO      = 1.0   
CONFIDENCE_MIN    = 0.5   

# तुझा GPS ज्या USB Port ला जोडला आहे त्याचे नाव इथे टाक:
# Windows साठी: 'COM3', 'COM5' वगैरे (Device Manager मध्ये तपासा)
# Linux/Mac साठी: '/dev/ttyUSB0'
GPS_PORT          = 'COM3' 
GPS_BAUDRATE      = 9600

CROWD_SOUND_FILE  = "crowd .mp3" 
FALLEN_SOUND_FILE = "alarm.wav"  
HELP_SOUND_FILE   = "help.mp3"

COLOR_NORMAL  = (0, 220, 0)     
COLOR_FALLEN  = (0, 0, 255)     
COLOR_SOS     = (255, 0, 255)   
COLOR_ALERT   = (0, 140, 255)   
COLOR_INFO    = (255, 255, 255) 
COLOR_LOC     = (0, 255, 255)   

# ग्लोबल व्हेरिएबल जे थ्रेड मधून अपडेट होईल
live_gps_location = "GPS"

# ────────────────────────────────────────────────────────────

def gps_tracker_thread():
    """हा फंक्शन बॅकग्राउंडमध्ये सतत GPS चे लोकेशन अपडेट करत राहील."""
    global live_gps_location
    try:
        ser = serial.Serial(GPS_PORT, baudrate=GPS_BAUDRATE, timeout=1)
        print(f"[✓] GPS Port '{GPS_PORT}' उघडला. सॅटेलाईट शोधत आहे...")
        
        while True:
            data = ser.readline().decode('ascii', errors='replace')
            # GPGGA किंवा GPRMC मध्ये अचूक लोकेशन असते
            if data.startswith('$GPGGA') or data.startswith('$GPRMC'):
                try:
                    msg = pynmea2.parse(data)
                    # जर सॅटेलाईट सिग्नल मिळाला असेल तर (latitude 0 नसतो)
                    if msg.latitude != 0.0 and msg.longitude != 0.0:
                        live_gps_location = f"📍 GPS: {msg.latitude:.5f}, {msg.longitude:.5f}"
                except pynmea2.ParseError:
                    pass # जर एखादा डेटा करप्ट आला तर इग्नोर करा
    except Exception as e:
        print(f"[✗] GPS एरर: {e}")
        live_gps_location = f"📍 GPS Error (Check Port {GPS_PORT})"

def draw_banner(frame, text, color, y_start=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + 50), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, y_start + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

def analyze_pose_and_boxes(results):
    person_data = []
    fallen_count = 0
    sos_count = 0

    boxes = results[0].boxes
    keypoints = results[0].keypoints

    if boxes is None or keypoints is None:
        return 0, 0, 0, []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        bw, bh = (x2 - x1), (y2 - y1)
        ratio = bw / bh if bh > 0 else 0
        is_fallen = ratio > FALLEN_RATIO
        
        if is_fallen:
            fallen_count += 1

        is_sos = False
        kpts = keypoints.data[i] 
        
        ls_x, ls_y, ls_c = kpts[5]
        rs_x, rs_y, rs_c = kpts[6]
        lw_x, lw_y, lw_c = kpts[9]
        rw_x, rw_y, rw_c = kpts[10]

        if all(c > CONFIDENCE_MIN for c in [ls_c, rs_c, lw_c, rw_c]):
            hands_up = (lw_y < ls_y) and (rw_y < rs_y)
            shoulder_width = abs(ls_x - rs_x)
            wrist_dist = abs(lw_x - rw_x)
            
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
    h, w = frame.shape[:2]

    # ── Live Hardware GPS Location Screen वर दाखवणे ──
    # global व्हेरिएबल मधून रिअल-टाइम डेटा मिळतो
    (loc_w, loc_h), _ = cv2.getTextSize(live_gps_location, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, live_gps_location, (w - loc_w - 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LOC, 2, cv2.LINE_AA)

    info_lines = [
        f"Persons Detected : {total}",
        f"Fallen Persons   : {fallen}",
        f"SOS Requests     : {sos}"
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 80 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)

    y_offset = h - 60
    
    if not is_crowd and fallen == 0 and sos == 0:
        draw_banner(frame, "Status: Normal", (40, 160, 40), y_start=y_offset)
        return

    if is_crowd:
        draw_banner(frame, "⚠ CROWD LIMIT EXCEEDED!", COLOR_ALERT, y_start=y_offset)
        y_offset -= 45
        
    if fallen > 0:
        draw_banner(frame, "⚠ EMERGENCY: FALLEN PERSON!", COLOR_FALLEN, y_start=y_offset)
        y_offset -= 45
        
    if sos > 0:
        draw_banner(frame, "🚨 SOS: SOMEONE NEEDS HELP!", COLOR_SOS, y_start=y_offset)

def main():
    print("=" * 45)
    print(" Public Hazurd + Crowd Anomaly Detection")
    print("=" * 45)

    # ── GPS Thread सुरू करणे ──
    # daemon=True म्हणजे जेव्हा मुख्य प्रोग्राम बंद होईल, तेव्हा हा थ्रेडही आपोआप बंद होईल
    gps_thread = threading.Thread(target=gps_tracker_thread, daemon=True)
    gps_thread.start()

    # ── Audio Setup ──
    pygame.mixer.init()
    ch_crowd = pygame.mixer.Channel(0)
    ch_fall  = pygame.mixer.Channel(1)
    ch_sos   = pygame.mixer.Channel(2)

    snd_crowd, snd_fall, snd_sos = None, None, None
    if os.path.exists(CROWD_SOUND_FILE): snd_crowd = pygame.mixer.Sound(CROWD_SOUND_FILE)
    if os.path.exists(FALLEN_SOUND_FILE): snd_fall = pygame.mixer.Sound(FALLEN_SOUND_FILE)
    if os.path.exists(HELP_SOUND_FILE): snd_sos = pygame.mixer.Sound(HELP_SOUND_FILE)
    
    # ── YOLO Model ──
    model = YOLO("yolov8n-pose.pt")
    print("[✓] YOLO Pose model लोड झाले!\n")
    print("[✓] mediapipe load!\n")
    print("[✓] openCV load!\n")

    print("Public Hazurd + Crowd Anomaly Detection Model [ START]")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        
        total, fallen, sos, person_data = analyze_pose_and_boxes(results)
        is_crowd = (total >= CROWD_THRESHOLD)

        if is_crowd and snd_crowd:
            if not ch_crowd.get_busy(): ch_crowd.play(snd_crowd, loops=-1)
        elif not is_crowd and ch_crowd.get_busy(): ch_crowd.stop()

        if fallen > 0 and snd_fall:
            if not ch_fall.get_busy(): ch_fall.play(snd_fall, loops=-1)
        elif fallen == 0 and ch_fall.get_busy(): ch_fall.stop()

        if sos > 0 and snd_sos:
            if not ch_sos.get_busy(): ch_sos.play(snd_sos, loops=-1)
        elif sos == 0 and ch_sos.get_busy(): ch_sos.stop()

        draw_banner(frame, "Public Hazurd + Crowd Anomaly Detection |  Press 'q' to Quit", (30, 30, 30))
        draw_detections(frame, person_data)
        
        # लोकेशन आणि इतर माहिती स्क्रीनवर टाकण्यासाठी
        overlay_stats(frame, total, fallen, sos, is_crowd)

        cv2.imshow("Public Hazurd + Crowd Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()