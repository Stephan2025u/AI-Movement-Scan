import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# Configurare MediaPipe stabilă
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    return 360-angle if angle > 180 else angle

st.set_page_config(page_title="Isokinetic AI Scan")
st.title("Scanare Biomecanică AI - Analiză Valg")

file = st.file_uploader("Încarcă Video (MP4/MOV)", type=['mp4', 'mov'])

if file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # Puncte cheie: Șold(23), Genunchi(25), Gleznă(27)
            h = [lm[23].x, lm[23].y]
            k = [lm[25].x, lm[25].y]
            a = [lm[27].x, lm[27].y]
            
            angle = get_angle(h, k, a)
            color = (0, 255, 0) if angle > 170 else (255, 0, 0) # Roșu dacă "pică" genunchiul
            
            mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(rgb, f"Unghi: {int(angle)} deg", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_placeholder.image(rgb, channels="RGB")
    cap.release()
