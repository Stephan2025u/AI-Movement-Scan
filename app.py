import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# Configurare MediaPipe
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

st.title("Isokinetic Movement Scan - AI")
st.write("Încarcă un video pentru analiza biomecanică a genunchiului.")

uploaded_file = st.file_input("Alege un video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesare imagine
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Puncte pentru unghi (Hip, Knee, Ankle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angle = calculate_angle(hip, knee, ankle)
                
                # Afisare unghi pe video
                cv2.putText(image, f"Valg: {int(angle)} deg", 
                           tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            st_frame.image(image)
            
    cap.release()
