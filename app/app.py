# app.py (Streamlit)
import streamlit as st
import cv2, os, time, tempfile, sqlite3, json, smtplib, requests, base64
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from email.mime.text import MIMEText

st.set_page_config(page_title="Human Detect→Confirm→Track→Alert", layout="wide")

MODEL_PATH_DEFAULT = "D:\\materials\\AI track\\materials\\DEEP LEARNING\\CV\\human_behaviour.yolov8\\Human_detect.pt"
OUTPUT_ROOT = "tracked_outputs"
DB_PATH = os.path.join(OUTPUT_ROOT, "records.db")

CONF_DEFAULT = 0.45
IOU_NMS_DEFAULT = 0.40
DEDUP_IOU_THRESH_DEFAULT = 0.50
CONFIRM_FRAMES_DEFAULT = 3
DIST_MATCH_THRESHOLD = 80
SMOOTH_ALPHA = 0.4
MAX_DISAPPEAR = 30
HIDE_ID_LABEL_DEFAULT = False
WEBHOOK_DEFAULT = ""
SMTP_DEFAULT = {"enabled": False, "server":"smtp.gmail.com","port":587,"user":"","pass":"","to":""}
os.makedirs(OUTPUT_ROOT, exist_ok=True)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        start_time TEXT,
        end_time TEXT,
        meta TEXT
    )''')
    conn.commit(); conn.close()
init_db()

def send_webhook(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print("Webhook error:", e)
        return False

def send_email(smtp_cfg, subject, body):
    if not smtp_cfg.get("enabled"): return False
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_cfg['user']
        msg['To'] = smtp_cfg['to']
        s = smtplib.SMTP(smtp_cfg['server'], smtp_cfg['port'], timeout=10)
        s.starttls()
        s.login(smtp_cfg['user'], smtp_cfg['pass'])
        s.sendmail(smtp_cfg['user'], [smtp_cfg['to']], msg.as_string())
        s.quit()
        return True
    except Exception as e:
        print("SMTP error:", e)
        return False

def iou_box(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, xa2-xa1) * max(0, ya2-ya1)
    area_b = max(0, xb2-xb1) * max(0, yb2-yb1)
    union = area_a + area_b - inter
    return inter / union if union>0 else 0.0
class Candidate:
    def __init__(self, bbox, centroid):
        self.bbox = bbox
        self.centroid = centroid
        self.count = 1
        self.last_seen = time.time()

class Track:
    def __init__(self, tid, bbox, centroid, smooth_alpha):
        self.id = tid
        self.bbox = bbox
        self.centroid = centroid
        self.smooth_alpha = smooth_alpha
        self.smoothed = centroid
        self.disappeared = 0
        self.trajectory = deque(maxlen=2048)
        self.trajectory.append(centroid)
    def update(self, bbox, centroid):
        sx, sy = self.smoothed
        cx, cy = centroid
        ax = self.smooth_alpha
        self.smoothed = (int(ax*cx + (1-ax)*sx), int(ax*cy + (1-ax)*sy))
        self.centroid = centroid
        self.bbox = bbox
        self.trajectory.append(self.smoothed)
        self.disappeared = 0

st.title("Robust Human Detect → Confirm → Track → Alert (Streamlit)")
st.sidebar.header("Model & thresholds")

MODEL_PATH = st.sidebar.text_input("Model path", MODEL_PATH_DEFAULT)
CONF_THRESHOLD = st.sidebar.slider("Confidence", 0.1, 0.9, CONF_DEFAULT, 0.05)
IOU_NMS = st.sidebar.slider("YOLO NMS IoU", 0.1, 0.7, IOU_NMS_DEFAULT, 0.05)
DEDUP_IOU = st.sidebar.slider("Dedup IoU", 0.2, 0.8, DEDUP_IOU_THRESH_DEFAULT, 0.05)
CONFIRM_FRAMES = st.sidebar.number_input("Confirm frames", 1, 10, CONFIRM_FRAMES_DEFAULT)
DIST_THRESHOLD = st.sidebar.number_input("Match distance (px)", 20, 400, DIST_MATCH_THRESHOLD)
SMOOTH_ALPHA = st.sidebar.slider("Smoothing alpha", 0.0, 1.0, SMOOTH_ALPHA, 0.05)
MAX_DISAPPEAR = st.sidebar.number_input("Max disappear frames", 1, 200, MAX_DISAPPEAR)
HIDE_ID_LABEL = st.sidebar.checkbox("Hide ID label", value=HIDE_ID_LABEL_DEFAULT)

st.sidebar.markdown("---")
st.sidebar.header("Alerts (webhook / SMTP)")
WEBHOOK_URL = st.sidebar.text_input("Webhook URL", value=WEBHOOK_DEFAULT)
use_smtp = st.sidebar.checkbox("Enable SMTP email", value=False)
smtp_cfg = SMTP_DEFAULT.copy()
if use_smtp:
    smtp_cfg["enabled"] = True
    smtp_cfg["server"] = st.sidebar.text_input("SMTP server", smtp_cfg["server"])
    smtp_cfg["port"] = st.sidebar.number_input("SMTP port", smtp_cfg["port"])
    smtp_cfg["user"] = st.sidebar.text_input("SMTP user", smtp_cfg["user"])
    smtp_cfg["pass"] = st.sidebar.text_input("SMTP pass", smtp_cfg["pass"], type="password")
    smtp_cfg["to"] = st.sidebar.text_input("Notify to", smtp_cfg["to"])

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
with col2:
    use_cam = st.checkbox("Use webcam (0)", value=False)

frame_window = st.image([])

@st.cache_resource(ttl=3600)
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Cannot load model: {e}")
    st.stop()

def process(cap, output_prefix="out"):
    next_id = 0
    tracks = {}
    candidates = []
    writer = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    recording = False
    rec_filename = None
    rec_start = None
    alert_sent = False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model(frame, conf=CONF_THRESHOLD, iou=IOU_NMS)[0]

        boxes = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else 1.0
                boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf))

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered = []
        for b in boxes:
            if all(iou_box(b[:4], fb[:4]) <= DEDUP_IOU for fb in filtered):
                filtered.append(b)
        boxes = filtered

        if (not recording) and len(boxes) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb, channels="RGB")
            if st.session_state.get("stop", False): break
            continue

        if (not recording) and len(boxes) > 0:
            recording = True
            rec_start = datetime.utcnow().isoformat(timespec='seconds') + "Z"
            rec_filename = f"{output_prefix}_{int(time.time())}.mp4"
            rec_path = os.path.join(OUTPUT_ROOT, rec_filename)
            writer = cv2.VideoWriter(rec_path, fourcc, fps, (width, height))
            payload = {"event":"human_detected","time":rec_start,"file":rec_filename,"count":len(boxes)}
            if WEBHOOK_URL: send_webhook(WEBHOOK_URL, payload)
            if smtp_cfg.get("enabled", False):
                send_email(smtp_cfg, "Human detected", f"Detected {len(boxes)} humans at {rec_start}")

        detections = []
        for (x1,y1,x2,y2,conf) in boxes:
            cx = (x1+x2)//2; cy = (y1+y2)//2
            detections.append(((x1,y1,x2,y2),(cx,cy),conf))

        used_tracks, used_dets = set(), set()
        if len(tracks)>0 and len(detections)>0:
            track_ids = list(tracks.keys())
            track_centroids = [tracks[tid].centroid for tid in track_ids]
            det_centroids = [d[1] for d in detections]
            D = np.linalg.norm(np.array(track_centroids)[:,None,:] - np.array(det_centroids)[None,:,:], axis=2)
            while True:
                i,j = np.unravel_index(np.argmin(D), D.shape)
                if np.isinf(D[i,j]): break
                if D[i,j] > DIST_THRESHOLD: break
                tid = track_ids[i]
                if tid in used_tracks or j in used_dets:
                    D[i,j] = np.inf; continue
                bbox, centroid, conf = detections[j]
                tracks[tid].update(bbox, centroid)
                used_tracks.add(tid); used_dets.add(j)
                D[i,:] = np.inf; D[:,j] = np.inf

        for idx, det in enumerate(detections):
            if idx in used_dets: continue
            bbox, centroid, conf = det
            matched = next((c for c in candidates if np.linalg.norm(np.array(c.centroid)-np.array(centroid)) < DIST_THRESHOLD/2), None)
            if matched:
                matched.count += 1; matched.bbox=bbox; matched.centroid=centroid; matched.last_seen=time.time()
                if matched.count >= CONFIRM_FRAMES:
                    tracks[next_id]=Track(next_id, bbox, centroid, SMOOTH_ALPHA)
                    next_id+=1; candidates.remove(matched)
            else:
                candidates.append(Candidate(bbox, centroid))

        candidates = [c for c in candidates if time.time()-c.last_seen<1.0]

        for tid in list(tracks.keys()):
            if tid not in used_tracks:
                tracks[tid].disappeared+=1
                if tracks[tid].disappeared>MAX_DISAPPEAR: del tracks[tid]

        display = frame.copy()
        # draw detections before tracking
        for (x1,y1,x2,y2,_) in boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(display,"Human Detected",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # draw tracks
        for tid, t in tracks.items():
            x1,y1,x2,y2 = map(int, t.bbox)
            color = (int(37+(tid*53)%200),int(99+(tid*97)%155),int(200-(tid*43)%150))
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            if not HIDE_ID_LABEL:
                cv2.putText(display,f"ID:{tid}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            pts = np.array(t.trajectory,dtype=np.int32)
            if len(pts)>1: cv2.polylines(display,[pts],False,color,2)

        fps_now = 1.0 / (time.time()-prev_time+1e-6)
        prev_time = time.time()
        cv2.putText(display,f"FPS:{fps_now:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        if writer is not None: writer.write(display)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        frame_window.image(rgb, channels="RGB")

        if st.session_state.get("stop", False): break

    if writer:
        writer.release()
        end_time = datetime.utcnow().isoformat(timespec='seconds') + "Z"
        meta = {"num_tracks":len(tracks),"trajectories":{tid:list(tr.trajectory) for tid,tr in tracks.items()}}
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO recordings (filename,start_time,end_time,meta) VALUES (?,?,?,?)",
                    (rec_filename, rec_start, end_time, json.dumps(meta)))
        conn.commit(); conn.close()
        st.success(f"Saved {rec_filename}")

        # Add download button
        video_path = os.path.join(OUTPUT_ROOT, rec_filename)
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        st.download_button("⬇️ Download Processed Video", video_bytes, file_name=rec_filename)

    cap.release()

# ------------------ Controls ------------------
if "stop" not in st.session_state: st.session_state["stop"]=False
c1,c2=st.columns([1,1])
with c1:
    if st.button("Start"):
        st.session_state["stop"]=False
        if uploaded:
            tf=tempfile.NamedTemporaryFile(delete=False)
            tf.write(uploaded.read()); tf.flush()
            cap=cv2.VideoCapture(tf.name)
            process(cap, output_prefix="upload_"+str(int(time.time())))
        elif use_cam:
            cap=cv2.VideoCapture(0)
            process(cap, output_prefix="cam_"+str(int(time.time())))
        else:
            st.warning("Upload a video or enable webcam.")
with c2:
    if st.button("Stop"):
        st.session_state["stop"]=True

# ------------------ Playback List ------------------
st.markdown("---")
st.header("Recordings")
conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
cur.execute("SELECT id,filename,start_time,end_time FROM recordings ORDER BY id DESC")
rows=cur.fetchall(); conn.close()
for rid,fname,stime,etime in rows:
    row_cols=st.columns([1,3,1,1])
    row_cols[0].write(f"ID {rid}")
    row_cols[1].write(f"{fname} — {stime} to {etime}")
    if row_cols[2].button("Play", key=f"play_{rid}"):
        path=os.path.join(OUTPUT_ROOT,fname)
        if os.path.exists(path): st.video(path)
        else: st.warning("File not found")
    if row_cols[3].button("Show meta", key=f"meta_{rid}"):
        conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
        cur.execute("SELECT meta FROM recordings WHERE id=?",(rid,))
        r=cur.fetchone(); conn.close()
        if r: st.json(json.loads(r[0]))
        else: st.warning("No meta")
