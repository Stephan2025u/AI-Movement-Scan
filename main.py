import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# Configurare rapidă
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("Isokinetic AI Scan")

uploaded_file = st.file_uploader("Încarcă Video", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Punctele pentru piciorul stâng: Șold(23), Genunchi(25), Glezna(27)
            lm = results.pose_landmarks.landmark
            h = [lm[23].x, lm[23].y]
            k = [lm[25].x, lm[25].y]
            a = [lm[27].x, lm[27].y]
            
            # Desenăm scheletul
            mp.solutions.drawing_utils.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.session_state['last_angle'] = "Detectat" # Indicator activitate

        frame_placeholder.image(rgb_frame, channels="RGB")
    cap.release()
else:
    st.info("Aștept încărcarea video-ului pentru analiză...")
