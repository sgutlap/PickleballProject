# detection.py
# Ball detection + trajectory tracking + shot classification.
# Person detection: OpenCV DNN MobileNet-SSD (auto-downloads model if missing).
# Falls back to HOG detector when DNN unavailable.

import cv2
import numpy as np
from typing import Dict, List
import time, os, urllib.request

TRACKERS = { }

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# URLs to download model (community-hosted). If these break, download them manually and place in backend/app/models/
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

PERSON_CLASS_ID = 15  # MobileNetSSD class id for 'person'

def ensure_dnn_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL):
        return True
    try:
        print("Downloading MobileNet-SSD model...")
        if not os.path.exists(PROTOTXT):
            urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT)
        if not os.path.exists(CAFFEMODEL):
            urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL)
        return True
    except Exception as e:
        print("Failed to download DNN model:", e)
        return False

def load_dnn_net():
    if ensure_dnn_model():
        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
            return net
        except Exception as e:
            print("Failed to load DNN net:", e)
            return None
    return None

# Ball detection as before
def detect_ball(frame_bgr):
    frame = frame_bgr.copy()
    h,w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for bright yellow/orange ball (tweak as needed)
    lower = np.array([5, 120, 150])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000: continue
        (x,y,wc,hc) = cv2.boundingRect(c)
        cx = x + wc/2
        cy = y + hc/2
        if area > best_area:
            best_area = area
            best = (int(cx), int(cy), int(max(wc,hc)/2))
    if best:
        return {'x': best[0], 'y': best[1], 'r': best[2], 'method': 'color'}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=3, maxRadius=60)
    if circles is not None:
        c = circles[0][0]
        return {'x': float(c[0]), 'y': float(c[1]), 'r': float(c[2]), 'method': 'hough'}
    return None

# Person detection using MobileNet-SSD DNN (preferred), fallback to HOG
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_players_dnn(frame_bgr, conf_threshold=0.5):
    net = load_dnn_net()
    if net is None:
        return None
    (h,w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    players = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        cls = int(detections[0,0,i,1])
        if conf > conf_threshold and cls == PERSON_CLASS_ID:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            cx = float((startX + endX) / 2.0)
            cy = float((startY + endY) / 2.0)
            players.append({'id': f'P{len(players)+1}', 'x': cx, 'y': cy, 'w': int(endX-startX), 'h': int(endY-startY), 'score': float(conf)})
    return players

def detect_players_hog(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects, weights = HOG.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
    players = []
    h,w = frame_bgr.shape[:2]
    for i, (x,y,ww,hh) in enumerate(rects):
        cx = x + ww/2
        cy = y + hh/2
        players.append({'id': f'P{i+1}', 'x': float(cx), 'y': float(cy), 'w': int(ww), 'h': int(hh), 'score': float(weights[i]) if i < len(weights) else 0.0})
    if not players:
        players = [{'id':'A','x':float(w*0.25),'y':float(h*0.75)},{'id':'B','x':float(w*0.75),'y':float(h*0.25)}]
    return players

def detect_players(frame_bgr):
    # prefer DNN, fallback to HOG heuristic
    players = detect_players_dnn(frame_bgr)
    if players is None:
        players = detect_players_hog(frame_bgr)
    return players

# shot classification (heuristic)
def classify_shot(traj: List[Dict]) -> str:
    if not traj or len(traj) < 3:
        return 'unknown'
    xs = np.array([p['x'] for p in traj])
    ys = np.array([p['y'] for p in traj])
    ts = np.array([p['t'] for p in traj])
    dt = (ts[-1] - ts[0]) / 1000.0 if ts[-1] != ts[0] else 1.0
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    speed = np.sqrt(dx*dx + dy*dy) / dt
    if speed < 200 and abs(dy) > 40 and dy < 0:
        return 'lob'
    if speed > 600 and abs(dx) > abs(dy):
        return 'drive'
    if speed < 220 and abs(dx) > abs(dy):
        return 'dink'
    return 'volley'

def add_traj(session_id, x, y):
    now = int(time.time()*1000)
    if session_id not in TRACKERS:
        TRACKERS[session_id] = []
    TRACKERS[session_id].append({'x':float(x),'y':float(y),'t':now})
    if len(TRACKERS[session_id]) > 40:
        TRACKERS[session_id] = TRACKERS[session_id][-40:]
    return TRACKERS[session_id]

def analyze_frame(frame_bytes: bytes, session_id: str = 'default'):
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Invalid image')
    h,w = img.shape[:2]
    ball = detect_ball(img)
    players = detect_players(img)
    traj = []
    shot = None
    tips = []
    if ball:
        traj = add_traj(session_id, ball['x'], ball['y'])
        shot = classify_shot(traj[-12:])
        if shot == 'dink':
            tips.append('Dink detected — move forward, keep paddle up, use soft touch.')
        elif shot == 'drive':
            tips.append('Drive detected — stay low, prepare for fast return and aim for angles.')
        elif shot == 'lob':
            tips.append(\"Lob detected — move back quickly and track the ball's highest point.\")
        elif shot == 'volley':
            tips.append('Volley detected — position racket early and control short swings.')
    else:
        tips.append('No ball detected in this frame — try a clearer camera angle or brighter ball.')
    return {
        'players': players,
        'ball': ball,
        'traj': traj[-20:],
        'shot': shot,
        'tips': tips,
    }
