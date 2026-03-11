import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# Inițializare MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

st.set_page_config(page_title="AI Biomechanics Scan", layout="wide")
st.title("Isokinetic Movement Scan - AI Analysis")
st.sidebar.title("Setări")

uploaded_file = st.sidebar.file_uploader("Încarcă video (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Salvăm video-ul temporar
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Conversie pentru MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Coordonate Picior Stâng (Punctele 23, 25, 27 în MediaPipe)
                hip = [landmarks[23].x, landmarks[23].y]
                knee = [landmarks[25].x, landmarks[25].y]
                ankle = [landmarks[27].x, landmarks[27].y]
                
                angle = calculate_angle(hip, knee, ankle)
                
                # Desenăm scheletul
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Afișăm unghiul lângă genunchi
                h, w, _ = image.shape
                pos = (int(knee[0] * w), int(knee[1] * h))
                cv2.putText(image, f"{int(angle)} deg", pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            st_frame.image(image, channels="RGB", use_container_width=True)
            
    cap.release()
    st.success("Analiză finalizată!")
else:
    st.info("Te rog încarcă un video din meniul lateral pentru a începe analiza.")
