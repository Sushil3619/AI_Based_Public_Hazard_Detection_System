"""
============================================================
  YOLO Crowd Anomaly Detection System
  - Person Detection (Bounding Box)
  - Fallen Person Detection (Width > Height)
  - Crowd Anomaly Alert (3+ लोक + 1 पडलेला = ALERT!)
============================================================
आवश्यक libraries install करा:
    pip install ultralytics opencv-python
============================================================
"""

import cv2
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
CROWD_THRESHOLD   = 3      # किती लोक असले की "गर्दी" म्हणायचं
FALLEN_RATIO      = 1.0    # width/height ratio > 1 → पडलेला माणूस
PERSON_CLASS_ID   = 0      # YOLO मध्ये class 0 = 'person'
CONFIDENCE_MIN    = 0.4    # किमान confidence score

# ── Colors (BGR format for OpenCV) ──────────────────────────
COLOR_NORMAL  = (0, 220, 0)     # हिरवा  → साधा माणूस
COLOR_FALLEN  = (0, 0, 255)     # लाल    → पडलेला माणूस
COLOR_ALERT   = (0, 140, 255)   # केशरी  → Alert banner
COLOR_INFO    = (255, 255, 255) # पांढरा → माहिती

# ────────────────────────────────────────────────────────────

def draw_banner(frame, text, color, y_start=0):
    """Screen वर आडवा banner दाखवतो."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + 50), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, y_start + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)


def analyze_persons(boxes):
    """
    प्रत्येक bounding box तपासतो:
      - उभा माणूस : height > width
      - पडलेला माणूस : width > height  (ratio > FALLEN_RATIO)
    Returns: (total_count, fallen_count, results_list)
    """
    results = []
    fallen_count = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])

        if cls != PERSON_CLASS_ID or conf < CONFIDENCE_MIN:
            continue

        bw = x2 - x1   # Bounding Box Width
        bh = y2 - y1   # Bounding Box Height
        ratio = bw / bh if bh > 0 else 0

        is_fallen = ratio > FALLEN_RATIO
        if is_fallen:
            fallen_count += 1

        results.append({
            "bbox"      : (x1, y1, x2, y2),
            "conf"      : conf,
            "is_fallen" : is_fallen,
            "ratio"     : ratio,
        })

    return len(results), fallen_count, results


def draw_detections(frame, results):
    """सर्व detected माणसांचे boxes आणि labels frame वर काढतो."""
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = COLOR_FALLEN if r["is_fallen"] else COLOR_NORMAL
        label = f"FALLEN! ({r['ratio']:.1f})" if r["is_fallen"] else f"Person ({r['conf']:.2f})"

        # Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def overlay_stats(frame, total, fallen, is_anomaly):
    """Screen च्या वरच्या कोपऱ्यात stats दाखवतो."""
    h, w = frame.shape[:2]

    # Top-left info box
    info_lines = [
        f"Persons Detected : {total}",
        f"Fallen Persons   : {fallen}",
        f"Crowd Threshold  : {CROWD_THRESHOLD}+",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 80 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)

    # ANOMALY ALERT banner (bottom)
    if is_anomaly:
        alert_text = "⚠  CROWD ANOMALY DETECTED — SOMEONE FALLEN IN CROWD!"
        draw_banner(frame, alert_text, COLOR_ALERT, y_start=h - 60)
    else:
        status = f"Status: {'CROWD DETECTED' if total >= CROWD_THRESHOLD else 'Normal'}  |  All persons standing"
        draw_banner(frame, status, (40, 160, 40), y_start=h - 60)


def main():
    print("=" * 60)
    print("  YOLO Crowd Anomaly Detection — सुरू होत आहे...")
    print("=" * 60)

    # YOLOv8 nano model लोड करा (पहिल्यांदा auto-download होईल)
    model = YOLO("yolov8n.pt")
    print("[✓] YOLO model लोड झाले!\n")

    # Webcam किंवा video file उघडा
    # Video file साठी: cap = cv2.VideoCapture("your_video.mp4")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[✗] Camera/Video उघडता आले नाही. Camera connected आहे का?")
        return

    print("[✓] Camera सुरू! 'Q' दाबा बंद करण्यासाठी.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[✗] Frame मिळाले नाही. Video संपले किंवा camera बंद झाला.")
            break

        # ── YOLO Inference ──────────────────────────────────
        results = model(frame, verbose=False)
        boxes   = results[0].boxes

        # ── Analysis ────────────────────────────────────────
        total, fallen, person_data = analyze_persons(boxes)

        # Anomaly: गर्दी + कोणी पडलेलं
        is_anomaly = (total >= CROWD_THRESHOLD) and (fallen > 0)

        # ── Visualization ───────────────────────────────────
        draw_banner(frame,
                    "YOLO Crowd Anomaly Detector  |  Press Q to Quit",
                    (30, 30, 30))
        draw_detections(frame, person_data)
        overlay_stats(frame, total, fallen, is_anomaly)

        # Terminal log
        if is_anomaly:
            print(f"[🚨 ALERT] गर्दीत माणूस पडला! Total={total}, Fallen={fallen}")
        elif fallen > 0:
            print(f"[⚠ WARNING] माणूस पडलेला आहे! (गर्दी नाही) Total={total}")

        cv2.imshow("Crowd Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[✓] प्रोग्राम बंद केला.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
