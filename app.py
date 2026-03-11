from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import cv2
import mediapipe as mp
import numpy as np
import json
import aiofiles
import io
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import shutil
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create upload directory
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = ROOT_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Create the main app
app = FastAPI(title="Kinema - Motion Analysis API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# MediaPipe Pose landmarks
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for angle calculations
LANDMARKS = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Pydantic Models
class SessionCreate(BaseModel):
    patient_name: str
    notes: Optional[str] = ""

class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_name: str
    notes: str = ""
    video_filename: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AnalysisResult(BaseModel):
    session_id: str
    frame_count: int
    fps: float
    duration: float
    landmarks_data: List[Dict[str, Any]]
    angles_data: Dict[str, List[float]]
    peak_angles: Dict[str, float]
    rom_data: Dict[str, Dict[str, float]]

# Utility functions
def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def calculate_horizontal_angle(p1, p2):
    """Calculate angle relative to horizontal"""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

def calculate_vertical_angle(p1, p2):
    """Calculate angle relative to vertical"""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = np.degrees(np.arctan2(dx, dy))
    return abs(angle)

def process_video_sync(video_path: str, session_id: str):
    """Process video and extract pose landmarks"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    landmarks_data = []
    angles_data = {
        'left_knee': [], 'right_knee': [],
        'left_hip': [], 'right_hip': [],
        'left_shoulder': [], 'right_shoulder': [],
        'left_elbow': [], 'right_elbow': [],
        'shoulder_alignment': [],
        'hip_tilt': [],
        'trunk_lean': [],
        'timestamps': []
    }
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            angles_data['timestamps'].append(round(timestamp, 3))
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Store landmarks for this frame
                frame_landmarks = {
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'landmarks': [
                        {'x': l.x, 'y': l.y, 'z': l.z, 'visibility': l.visibility}
                        for l in lm
                    ]
                }
                landmarks_data.append(frame_landmarks)
                
                # Calculate angles
                # Left Knee (Hip-Knee-Ankle)
                left_knee_angle = calculate_angle(
                    lm[LANDMARKS['left_hip']],
                    lm[LANDMARKS['left_knee']],
                    lm[LANDMARKS['left_ankle']]
                )
                angles_data['left_knee'].append(round(left_knee_angle, 1))
                
                # Right Knee
                right_knee_angle = calculate_angle(
                    lm[LANDMARKS['right_hip']],
                    lm[LANDMARKS['right_knee']],
                    lm[LANDMARKS['right_ankle']]
                )
                angles_data['right_knee'].append(round(right_knee_angle, 1))
                
                # Left Hip (Shoulder-Hip-Knee)
                left_hip_angle = calculate_angle(
                    lm[LANDMARKS['left_shoulder']],
                    lm[LANDMARKS['left_hip']],
                    lm[LANDMARKS['left_knee']]
                )
                angles_data['left_hip'].append(round(left_hip_angle, 1))
                
                # Right Hip
                right_hip_angle = calculate_angle(
                    lm[LANDMARKS['right_shoulder']],
                    lm[LANDMARKS['right_hip']],
                    lm[LANDMARKS['right_knee']]
                )
                angles_data['right_hip'].append(round(right_hip_angle, 1))
                
                # Left Shoulder (Elbow-Shoulder-Hip)
                left_shoulder_angle = calculate_angle(
                    lm[LANDMARKS['left_elbow']],
                    lm[LANDMARKS['left_shoulder']],
                    lm[LANDMARKS['left_hip']]
                )
                angles_data['left_shoulder'].append(round(left_shoulder_angle, 1))
                
                # Right Shoulder
                right_shoulder_angle = calculate_angle(
                    lm[LANDMARKS['right_elbow']],
                    lm[LANDMARKS['right_shoulder']],
                    lm[LANDMARKS['right_hip']]
                )
                angles_data['right_shoulder'].append(round(right_shoulder_angle, 1))
                
                # Left Elbow
                left_elbow_angle = calculate_angle(
                    lm[LANDMARKS['left_shoulder']],
                    lm[LANDMARKS['left_elbow']],
                    lm[LANDMARKS['left_wrist']]
                )
                angles_data['left_elbow'].append(round(left_elbow_angle, 1))
                
                # Right Elbow
                right_elbow_angle = calculate_angle(
                    lm[LANDMARKS['right_shoulder']],
                    lm[LANDMARKS['right_elbow']],
                    lm[LANDMARKS['right_wrist']]
                )
                angles_data['right_elbow'].append(round(right_elbow_angle, 1))
                
                # Shoulder Alignment (angle between shoulders relative to horizontal)
                shoulder_align = calculate_horizontal_angle(
                    lm[LANDMARKS['left_shoulder']],
                    lm[LANDMARKS['right_shoulder']]
                )
                angles_data['shoulder_alignment'].append(round(shoulder_align, 1))
                
                # Hip Tilt (angle between hips relative to horizontal)
                hip_tilt = calculate_horizontal_angle(
                    lm[LANDMARKS['left_hip']],
                    lm[LANDMARKS['right_hip']]
                )
                angles_data['hip_tilt'].append(round(hip_tilt, 1))
                
                # Trunk Lean (vertical alignment of spine - mid shoulder to mid hip)
                mid_shoulder_x = (lm[LANDMARKS['left_shoulder']].x + lm[LANDMARKS['right_shoulder']].x) / 2
                mid_shoulder_y = (lm[LANDMARKS['left_shoulder']].y + lm[LANDMARKS['right_shoulder']].y) / 2
                mid_hip_x = (lm[LANDMARKS['left_hip']].x + lm[LANDMARKS['right_hip']].x) / 2
                mid_hip_y = (lm[LANDMARKS['left_hip']].y + lm[LANDMARKS['right_hip']].y) / 2
                
                trunk_angle = np.degrees(np.arctan2(mid_shoulder_x - mid_hip_x, mid_hip_y - mid_shoulder_y))
                angles_data['trunk_lean'].append(round(abs(trunk_angle), 1))
            else:
                # No landmarks detected - add None values
                for key in angles_data:
                    if key != 'timestamps':
                        angles_data[key].append(None)
            
            frame_idx += 1
    
    cap.release()
    
    # Calculate peak angles and ROM
    peak_angles = {}
    rom_data = {}
    
    for key in angles_data:
        if key != 'timestamps':
            valid_values = [v for v in angles_data[key] if v is not None]
            if valid_values:
                peak_angles[key] = max(valid_values)
                rom_data[key] = {
                    'min': min(valid_values),
                    'max': max(valid_values),
                    'range': max(valid_values) - min(valid_values),
                    'mean': round(np.mean(valid_values), 1)
                }
    
    return {
        'session_id': session_id,
        'frame_count': frame_count,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'landmarks_data': landmarks_data,
        'angles_data': angles_data,
        'peak_angles': peak_angles,
        'rom_data': rom_data
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Kinema API - Full Body Motion Analysis"}

# Session CRUD
@api_router.post("/sessions", response_model=Session)
async def create_session(session_data: SessionCreate):
    session = Session(
        patient_name=session_data.patient_name,
        notes=session_data.notes or ""
    )
    doc = session.model_dump()
    await db.sessions.insert_one(doc)
    return session

@api_router.get("/sessions", response_model=List[Session])
async def get_sessions():
    sessions = await db.sessions.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return sessions

@api_router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@api_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    result = await db.sessions.delete_one({"id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete associated files
    for ext in ['mp4', 'mov', 'avi', 'webm']:
        video_path = UPLOAD_DIR / f"{session_id}.{ext}"
        if video_path.exists():
            video_path.unlink()
    
    return {"message": "Session deleted"}

@api_router.put("/sessions/{session_id}")
async def update_session(session_id: str, session_data: SessionCreate):
    update_doc = {
        "patient_name": session_data.patient_name,
        "notes": session_data.notes or "",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    result = await db.sessions.update_one({"id": session_id}, {"$set": update_doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    return session

# Video Upload and Analysis
@api_router.post("/sessions/{session_id}/upload")
async def upload_video(session_id: str, file: UploadFile = File(...)):
    # Verify session exists
    session = await db.sessions.find_one({"id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate file type
    allowed_types = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {allowed_types}")
    
    # Save file
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
    video_path = UPLOAD_DIR / f"{session_id}.{ext}"
    
    async with aiofiles.open(video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Update session with filename
    await db.sessions.update_one(
        {"id": session_id},
        {"$set": {
            "video_filename": f"{session_id}.{ext}",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Video uploaded successfully", "filename": f"{session_id}.{ext}"}

@api_router.post("/sessions/{session_id}/analyze")
async def analyze_video(session_id: str, background_tasks: BackgroundTasks):
    # Get session
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.get('video_filename'):
        raise HTTPException(status_code=400, detail="No video uploaded for this session")
    
    video_path = UPLOAD_DIR / session['video_filename']
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Process video
    try:
        analysis_result = process_video_sync(str(video_path), session_id)
        
        # Store analysis data in session
        await db.sessions.update_one(
            {"id": session_id},
            {"$set": {
                "analysis_data": analysis_result,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return analysis_result
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/sessions/{session_id}/video")
async def get_video(session_id: str):
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.get('video_filename'):
        raise HTTPException(status_code=404, detail="No video for this session")
    
    video_path = UPLOAD_DIR / session['video_filename']
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=session['video_filename']
    )

# Export endpoints
@api_router.get("/sessions/{session_id}/export/csv")
async def export_csv(session_id: str):
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.get('analysis_data'):
        raise HTTPException(status_code=400, detail="No analysis data available")
    
    analysis = session['analysis_data']
    
    # Create Excel workbook
    wb = Workbook()
    
    # Angles sheet
    ws_angles = wb.active
    ws_angles.title = "Joint Angles"
    
    # Headers
    headers = ['Timestamp', 'Left Knee', 'Right Knee', 'Left Hip', 'Right Hip', 
               'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
               'Shoulder Alignment', 'Hip Tilt', 'Trunk Lean']
    ws_angles.append(headers)
    
    # Data
    angles = analysis['angles_data']
    for i, ts in enumerate(angles.get('timestamps', [])):
        row = [
            ts,
            angles.get('left_knee', [None])[i] if i < len(angles.get('left_knee', [])) else None,
            angles.get('right_knee', [None])[i] if i < len(angles.get('right_knee', [])) else None,
            angles.get('left_hip', [None])[i] if i < len(angles.get('left_hip', [])) else None,
            angles.get('right_hip', [None])[i] if i < len(angles.get('right_hip', [])) else None,
            angles.get('left_shoulder', [None])[i] if i < len(angles.get('left_shoulder', [])) else None,
            angles.get('right_shoulder', [None])[i] if i < len(angles.get('right_shoulder', [])) else None,
            angles.get('left_elbow', [None])[i] if i < len(angles.get('left_elbow', [])) else None,
            angles.get('right_elbow', [None])[i] if i < len(angles.get('right_elbow', [])) else None,
            angles.get('shoulder_alignment', [None])[i] if i < len(angles.get('shoulder_alignment', [])) else None,
            angles.get('hip_tilt', [None])[i] if i < len(angles.get('hip_tilt', [])) else None,
            angles.get('trunk_lean', [None])[i] if i < len(angles.get('trunk_lean', [])) else None,
        ]
        ws_angles.append(row)
    
    # ROM Summary sheet
    ws_rom = wb.create_sheet("ROM Summary")
    ws_rom.append(['Joint', 'Min (°)', 'Max (°)', 'Range (°)', 'Mean (°)'])
    
    rom = analysis.get('rom_data', {})
    for joint, data in rom.items():
        ws_rom.append([
            joint.replace('_', ' ').title(),
            data.get('min'),
            data.get('max'),
            data.get('range'),
            data.get('mean')
        ])
    
    # Save to bytes
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    filename = f"kinema_analysis_{session['patient_name']}_{session_id[:8]}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@api_router.get("/sessions/{session_id}/export/pdf")
async def export_pdf(session_id: str):
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.get('analysis_data'):
        raise HTTPException(status_code=400, detail="No analysis data available")
    
    analysis = session['analysis_data']
    
    # Create PDF
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor('#09090B')
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#3B82F6')
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("KINEMA Motion Analysis Report", title_style))
    elements.append(Spacer(1, 10))
    
    # Patient Info
    elements.append(Paragraph(f"Patient: {session['patient_name']}", subtitle_style))
    elements.append(Paragraph(f"Date: {session['created_at'][:10]}", styles['Normal']))
    elements.append(Paragraph(f"Duration: {analysis['duration']:.2f} seconds | Frames: {analysis['frame_count']} | FPS: {analysis['fps']:.1f}", styles['Normal']))
    
    if session.get('notes'):
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Notes: {session['notes']}", styles['Normal']))
    
    elements.append(Spacer(1, 20))
    
    # ROM Summary Table
    elements.append(Paragraph("Range of Motion Summary", subtitle_style))
    
    rom = analysis.get('rom_data', {})
    table_data = [['Joint', 'Min (°)', 'Max (°)', 'Range (°)', 'Mean (°)']]
    
    for joint, data in rom.items():
        table_data.append([
            joint.replace('_', ' ').title(),
            f"{data.get('min', 'N/A')}",
            f"{data.get('max', 'N/A')}",
            f"{data.get('range', 'N/A')}",
            f"{data.get('mean', 'N/A')}"
        ])
    
    table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F4F4F5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D4D4D8')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    elements.append(table)
    
    elements.append(Spacer(1, 20))
    
    # Peak Angles
    elements.append(Paragraph("Peak Angles Detected", subtitle_style))
    
    peak = analysis.get('peak_angles', {})
    peak_data = [['Joint', 'Peak Angle (°)']]
    for joint, angle in peak.items():
        peak_data.append([joint.replace('_', ' ').title(), f"{angle}"])
    
    peak_table = Table(peak_data, colWidths=[3*inch, 2*inch])
    peak_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EF4444')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D4D4D8')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    elements.append(peak_table)
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Generated by Kinema - Precision Motion Analysis", styles['Normal']))
    
    doc.build(elements)
    output.seek(0)
    
    filename = f"kinema_report_{session['patient_name']}_{session_id[:8]}.pdf"
    
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Comparison endpoint
@api_router.post("/compare")
async def compare_sessions(session_ids: List[str]):
    if len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sessions to compare")
    
    sessions = []
    for sid in session_ids:
        session = await db.sessions.find_one({"id": sid}, {"_id": 0})
        if session and session.get('analysis_data'):
            sessions.append(session)
    
    if len(sessions) < 2:
        raise HTTPException(status_code=400, detail="Not enough sessions with analysis data")
    
    return {
        "sessions": sessions,
        "comparison_type": "multi_session"
    }

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
