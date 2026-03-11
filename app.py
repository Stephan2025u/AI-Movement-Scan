import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Verificăm dacă mediapipe poate fi inițializat
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"Eroare la inițializarea AI: {e}")

st.set_page_config(page_title="Kineto AI Scan", layout="centered")
st.title("Biomecanică AI: Analiză Valg Genunchi")

st.markdown("""
Acest tool analizează unghiul genunchiului în timpul săriturii (Drop Jump) pentru a identifica riscul de accidentare LIA.
""")

uploaded_file = st.file_uploader("Încarcă video", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Procesare minimă pentru viteză și stabilitate
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Desenăm scheletul simplificat
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Calcul unghi (Exemplu pentru piciorul stâng)
                try:
                    lm = results.pose_landmarks.landmark
                    # Puncte: Hip(23), Knee(25), Ankle(27)
                    h = [lm[23].x, lm[23].y]
                    k = [lm[25].x, lm[25].y]
                    a = [lm[27].x, lm[27].y]
                    
                    # Logică simplă de calcul unghi
                    ba = np.array(h) - np.array(k)
                    bc = np.array(a) - np.array(k)
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(cosine_angle))
                    
                    cv2.putText(rgb_frame, f"Unghi: {int(angle)} deg", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass

            frame_placeholder.image(rgb_frame, channels="RGB")
    
    cap.release()
