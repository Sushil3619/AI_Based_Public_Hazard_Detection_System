import cv2
import mediapipe as mp

# 1. MediaPipe Pose सेटअप
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. कॅमेरा सुरु करणे
cap = cv2.VideoCapture(0)
print("Smart Rescue System सुरु होत आहे... (बंद करण्यासाठी 'q' दाबा)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("कॅमेरा उघडता आला नाही.")
        break
        
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- Bounding Box चे को-ऑर्डिनेट्स शोधणे ---
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            
        pad = 20
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
        
        # --- शरीराच्या महत्त्वाच्या सांध्यांचे (Landmarks) पॉइंट्स ---
        # Fall Detection साठी (नाक आणि कंबर)
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        
        # SOS Gesture साठी (खांदे आणि मनगट)
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        
        # --- सर्व अटी (Conditions) ठरवणे ---
        is_fallen = nose_y > avg_hip_y - 0.1
        hands_up = (left_wrist_y < left_shoulder_y) and (right_wrist_y < right_shoulder_y)
        wrists_crossed = abs(left_wrist_x - right_wrist_x) < 0.1
        is_x_sign = hands_up and wrists_crossed
        
        # --- PRIORITY LOGIC (प्राधान्यक्रम) ---
        
        # 1. सर्वात मोठी Emergency: व्यक्ती खाली पडली आहे
        if is_fallen:
            box_color = (0, 0, 255) # लाल (Red)
            alert_text = "🚨 ALERT: Person Fallen!"
            
        # 2. दुसरी Emergency: 'X' साईन बनवली आहे
        elif is_x_sign:
            box_color = (0, 0, 255) # लाल (Red)
            alert_text = "🚨 SOS: 'X' Sign Detected!"
            
        # 3. तिसरी Emergency: दोन्ही हात वर केले आहेत
        elif hands_up:
            box_color = (0, 165, 255) # केशरी (Orange)
            alert_text = "⚠️ SOS: Hands Raised!"
            
        # 4. अन्यथा: सर्व सुरक्षित (Safe) आहे
        else:
            box_color = (0, 255, 0) # हिरवा (Green)
            alert_text = "✅ Safe: Normal"
            
        # --- Bounding Box आणि Text ड्रॉ करणे ---
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
        cv2.putText(frame, alert_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                        
    # फ्रेम दाखवणे
    cv2.imshow('Smart Rescue System (Priority Logic)', frame)
    
    # 'q' दाबलं की बाहेर पडणे
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()