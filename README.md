# AI Human Detection & Tracking System  
High-Accuracy Person Identification • Multi-Target Tracking • Automated Recording & Alerts

## Overview
This project implements a high-precision artificial intelligence system for **real-time human detection and persistent multi-target tracking**.  
The pipeline is built using a custom-trained **YOLO model** integrated with a **Streamlit interface**, and is designed for professional-grade security, monitoring, and intelligent camera applications.

The system delivers **97%+ detection accuracy**, performs **multi-frame verification** before confirming a target, and maintains **stable tracking IDs** for every individual.  
It automatically records events, logs structured metadata, and supports **Webhook** and **Email (SMTP)** notifications.

This framework is engineered for real deployments such as security cameras, access-control systems, restricted-area monitoring, and future AI-driven robotic camera automation.

<img width="584" height="1015" alt="live (3)" src="https://github.com/user-attachments/assets/0c870c7f-8187-4ea8-af38-86905781dd89" />

---

## Key Features
- High-accuracy human detection using a custom YOLO model  
- Multi-frame confirmation to reduce false positives  
- Persistent multi-target tracking with unique ID assignment  
- Automatic recording on first verified detection  
- Exported processed video with bounding boxes & trajectories  
- Webhook alert with real-time JSON event notifications  
- Email alert (SMTP) with timestamp and recording reference  
- Local SQLite database for event logging and metadata storage  
- Compatible with live camera feeds, USB webcam, and uploaded video files  
- Clean, reliable Streamlit-based interface

---

## System Architecture
1. **Frame Acquisition:** Camera feed or uploaded video is streamed in real time.  
2. **YOLO Inference:** Human detections are extracted with confidence scores.  
3. **Verification Stage:** Detections must appear across several frames before becoming confirmed targets.  
4. **Track Management:** Each confirmed person receives a persistent tracking ID.  
5. **Trajectory Rendering:** Tracks are updated, smoothed, and drawn on the output frames.  
6. **Event Logging & Alerts:** Video recordings and metadata are saved, and Webhook/SMTP alerts can be triggered.  

---

## Folder Structure
```
app.py                   # Main application
tracked_outputs/         # Processed videos and metadata
    records.db           # SQLite event database
models/                  # (Optional) YOLO model storage
requirements.txt         # Python dependencies
README.md                # Documentation
```

---

## Installation
### Clone the Repository
```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

### Create & Activate Virtual Environment
```bash
python -m venv venv
```
Activate:
- Windows: `venv\Scripts\activate`  
- Linux/Mac: `source venv/bin/activate`

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application
1. Place your YOLO model (e.g., `Human_detect.pt`) in the project folder.  
2. Set the model path inside the Streamlit sidebar.  
3. Launch the app:
```bash
streamlit run app.py
```

Access it via the URL shown (usually http://localhost:8501).

---

## Alerts & Notifications

### Webhook
A detection triggers a JSON event:
```json
{
  "event": "human_detected",
  "timestamp": "....",
  "video": "record_001.mp4"
}
```

### Email (SMTP)
Configured through the Streamlit sidebar for secure email alerting with automatic TLS.

---

## Database Logging
All events are stored in:
```
tracked_outputs/records.db
```
Each entry contains:
- Video name  
- Start/End timestamps  
- Number of detections  
- Alert status  

---

## Configuration Options
- Model file path  
- Detection confidence threshold  
- Confirmation frame count  
- Maximum disappear frames  
- Recording enable/disable  
- Webhook URL  
- SMTP configuration  

---

## Future Enhancements
- Servo-based camera control (auto-follow specific target)  
- Identity-aware tracking via facial recognition  
- Multi-camera distributed monitoring  
- Cloud dashboard integration  
- Behavioral analytics (loitering, line-crossing, intrusion zones)

---

## License
MIT License.

