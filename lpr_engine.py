import os
# P-C: Limit ALL math/parallel libraries to 1 thread to cut PaddleOCR peak CPU.
# PaddleOCR runs 3 neural networks (det/cls/rec) and ignores OMP unless
# Paddle-specific env vars are also set. Must be set BEFORE any import.
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP (ORT + numpy + paddle)
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS (paddle det/rec models)
os.environ["PADDLE_NUM_THREADS"] = "1"       # Paddle internal thread pool
os.environ["CPU_NUM"] = "1"                  # Paddle CPU executor threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # NumExpr (numpy accelerator)
os.environ["KMP_BLOCKTIME"] = "0"            # OpenMP thread blocktime: go to sleep immediately
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"    # OpenMP wait policy: yield cores immediately
os.environ["TBB_MAX_ALLOWED_NUM_THREADS"] = "1" # Intel TBB (OpenVINO default scheduler) max threads
# Disable FFMPEG multi-threaded decoding globally to bypass pthread_frame.c crash
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;stimeout;5000000"
# Disable MKLDNN globally to bypass oneDNN ConvertPirAttribute2RuntimeAttribute bug
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
import cv2
import numpy as np
import time
import sqlite3
import datetime
import threading
import queue
import re
import json
import urllib.parse
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import TCPServer
from ultralytics import YOLO
from paddleocr import PaddleOCR
import supervision as sv  # Roboflow Supervision: PolygonZone gate filtering

# Configure Paths
PUBLIC_DIR = r"D:\AntiGravity\ai camera-gate\public"
DATA_DIR = r"D:\AntiGravity\lpr_data"
DB_PATH = os.path.join(DATA_DIR, "lpr.db")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
FULLS_DIR = os.path.join(DATA_DIR, "fulls")
ZONE_CONFIG_PATH = os.path.join(DATA_DIR, "zone_config.json")

# Ensure directories exist
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(FULLS_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Global variables for sharing frames between threads
latest_display_frames = {} # Key: camera_id (int), Value: JPEG bytes (compressed frame)
frame_lock = threading.Lock()
video_capture_lock = threading.Lock()

# ── Gate Zone Polygons (module-level for HTTP handler access) ─────────────────
# Coordinates are in full-frame pixels (1920×1080).
# Loaded from zone_config.json on startup; updated live via POST /api/zones.
DEFAULT_ZONE_POLYGONS = {
    "1": [[320, 80], [1800, 80], [1800, 1000], [320, 1000]],
    "3": [[250, 30], [1820, 30], [1820, 1050], [250, 1050]],
    "6": [[100, 50], [1820, 50], [1820, 1000], [100, 1000]],
}
GATE_ZONE_POLYGONS = {}   # int cam_id -> np.array polygon
gate_zones = {}            # int cam_id -> sv.PolygonZone
zone_config_lock = threading.Lock()

def load_zone_config():
    """Load zone polygons from zone_config.json; return DEFAULT_ZONE_POLYGONS if missing."""
    if os.path.exists(ZONE_CONFIG_PATH):
        try:
            with open(ZONE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[SYSTEM] zone_config.json load error: {e}. Using defaults.")
    return DEFAULT_ZONE_POLYGONS.copy()

def save_zone_config(zones_dict):
    """Persist zones_dict (str cam_id -> list of [x,y]) to zone_config.json."""
    try:
        with open(ZONE_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(zones_dict, f, indent=2)
    except Exception as e:
        print(f"[SYSTEM] zone_config.json save error: {e}")

def rebuild_gate_zones():
    """Rebuild sv.PolygonZone objects from the current GATE_ZONE_POLYGONS global."""
    global gate_zones
    new_zones = {}
    for cam_id, poly in GATE_ZONE_POLYGONS.items():
        new_zones[cam_id] = sv.PolygonZone(
            polygon=poly,
            triggering_anchors=[sv.Position.CENTER]
        )
    gate_zones = new_zones
    print(f"[SYSTEM] PolygonZone gate filters rebuilt for cameras: {list(gate_zones.keys())}")

# Load Configuration from config.json
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
def load_config():
    default_config = {
        "rtsp_url_gate": "rtsp://admin:%40fsps087525402@192.168.1.131:554/chID=1&streamType=main",
        "rtsp_url_gate_002": "rtsp://admin:%40Fsps087525402@192.168.1.138:554/chID=5&streamType=main",
        "web_username": "admin",
        "web_password_default": "3762828"
    }
    if not os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
        except Exception:
            pass
        return default_config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_config

config_data = load_config()
RTSP_URL = config_data.get("rtsp_url_gate")
RTSP_URL_002 = config_data.get("rtsp_url_gate_002")

# Sanitize and format RTSP URL (URL-encode password if needed, remove whitespaces)
def sanitize_rtsp_url(url):
    url = "".join(url.split())
    if not url.startswith("rtsp://"):
        return url
    try:
        rest = url[7:]
        if '@' in rest:
            creds, host = rest.rsplit('@', 1)
            if ':' in creds:
                user, password = creds.split(':', 1)
                decoded_password = urllib.parse.unquote(password)
                encoded_password = urllib.parse.quote(decoded_password)
                return f"rtsp://{user}:{encoded_password}@{host}"
    except Exception as e:
        print(f"Error sanitizing RTSP URL: {e}")
    return url

# Database Setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Enable Write-Ahead Logging (WAL) and set synchronous write to NORMAL for high concurrency performance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_number TEXT NOT NULL,
            confidence REAL,
            crop_image_path TEXT,
            full_image_path TEXT
        )
    """)
    # Add vehicle_type column if it doesn't exist (DB Migration)
    try:
        cursor.execute("ALTER TABLE detections ADD COLUMN vehicle_type TEXT DEFAULT 'CAR'")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE detections ADD COLUMN camera_id INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
        
    # Retroactively classify existing plates (bidirectional migration)
    cursor.execute("SELECT id, plate_number, vehicle_type FROM detections")
    existing_rows = cursor.fetchall()
    updates = []
    for row_id, plate, current_type in existing_rows:
        v_type = classify_vehicle_type(plate)
        if v_type != current_type:
            updates.append((v_type, row_id))
    if updates:
        cursor.executemany("UPDATE detections SET vehicle_type = ? WHERE id = ?", updates)
        conn.commit()
        print(f"Retroactively updated classification for {len(updates)} detections.")
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            plate_number TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            plate_number TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            action TEXT DEFAULT 'ENTRY',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(detection_id) REFERENCES detections(id)
        )
    """)
    try:
        cursor.execute("ALTER TABLE alerts ADD COLUMN camera_id INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE alerts ADD COLUMN action TEXT DEFAULT 'ENTRY'")
    except sqlite3.OperationalError:
        pass
    # Pre-populate watchlist with test data if empty
    cursor.execute("SELECT COUNT(*) FROM watchlist")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO watchlist (plate_number, category, description) 
            VALUES (?, ?, ?)
        """, [
            ('TEST1234', 'BLACKLIST', '測試黑名單車輛'),
            ('VIP8888', 'VIP', '校長專車'),
            ('BJU2379', 'VIP', '測試VIP車輛')
        ])
        
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add is_active column if it doesn't exist (DB Migration)
    try:
        cursor.execute("ALTER TABLE cameras ADD COLUMN is_active INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
        
    cursor.execute("SELECT COUNT(*) FROM cameras")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO cameras (id, name, rtsp_url, is_active) VALUES (1, '學校大門002', ?, 1)", (RTSP_URL_002,))
        cursor.execute("INSERT INTO cameras (id, name, rtsp_url, is_active) VALUES (3, '學校大門', ?, 1)", (RTSP_URL,))
    else:
        # Check if at least one camera is active, if not activate the first one
        cursor.execute("SELECT COUNT(*) FROM cameras WHERE is_active = 1")
        if cursor.fetchone()[0] == 0:
            cursor.execute("UPDATE cameras SET is_active = 1 WHERE id = (SELECT id FROM cameras LIMIT 1)")
        
    # Initialize Telegram Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('tg_enabled', '0')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('tg_bot_token', '')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('tg_chat_id', '')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('web_username', 'admin')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('web_password', '3762828')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('retention_days', '30')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('offline_threshold_minutes', '5')")

# go2rtc global processes and cache
go2rtc_process = None
last_active_cameras_cache = None

def generate_go2rtc_yaml():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, rtsp_url FROM cameras WHERE is_active = 1")
    rows = cursor.fetchall()
    conn.close()
    
    yaml_lines = [
        "api:",
        "  listen: \":1984\"",
        "",
        "rtsp:",
        "  listen: \":8554\"",
        "",
        "ffmpeg:",
        "  bin: \"D:/AntiGravity/ffmpeg.exe\"",
        "",
        "streams:"
    ]
    
    for cam_id, url in rows:
        decoded_url = urllib.parse.unquote(url)
        if "192.168.1.138" in decoded_url:
            source_str = f"exec:D:/AntiGravity/ffmpeg.exe -analyzeduration 10000000 -probesize 10000000 -i {decoded_url} -c copy -an -f rtsp {{output}}"
        else:
            source_str = decoded_url
            
        yaml_lines.append(f"  camera{cam_id}: \"{source_str}\"")
        
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(engine_dir, "go2rtc.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    print(f"[SYSTEM] Generated go2rtc.yaml with {len(rows)} streams.")

def start_go2rtc():
    global go2rtc_process
    generate_go2rtc_yaml()
    
    try:
        subprocess.run("taskkill /f /im go2rtc.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(0.5)
    
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(engine_dir, "go2rtc.exe")
    
    if not os.path.exists(exe_path):
        print(f"[SYSTEM] Warning: go2rtc.exe not found at {exe_path}!")
        return
        
    print("[SYSTEM] Starting go2rtc background service...")
    try:
        log_dir = r"D:\AntiGravity\lpr_data"
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, "go2rtc.log"), "w", encoding="utf-8")
        
        go2rtc_process = subprocess.Popen(
            [exe_path],
            cwd=engine_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        time.sleep(2.0)
        print("[SYSTEM] go2rtc started successfully.")
    except Exception as e:
        print(f"[SYSTEM] Failed to start go2rtc: {e}")

def stop_go2rtc():
    global go2rtc_process
    if go2rtc_process:
        print("[SYSTEM] Terminating go2rtc service...")
        go2rtc_process.terminate()
        try:
            go2rtc_process.wait(timeout=2)
        except Exception:
            try:
                go2rtc_process.kill()
            except Exception:
                pass
        go2rtc_process = None

def sync_active_readers():
    global active_readers, active_camera_names, last_active_cameras_cache
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, rtsp_url FROM cameras WHERE is_active = 1")
    active_cameras = cursor.fetchall()
    conn.close()

    # Detect modifications to active cameras
    current_cache = {r[0]: sanitize_rtsp_url(r[2]) for r in active_cameras}
    if last_active_cameras_cache is None or last_active_cameras_cache != current_cache:
        print("[SYSTEM] Camera configuration changed. Restarting go2rtc proxy...")
        last_active_cameras_cache = current_cache
        start_go2rtc()

    active_ids = {r[0] for r in active_cameras}

    # Stop readers that are no longer active first
    for cam_id in list(active_readers.keys()):
        if cam_id not in active_ids:
            print(f"[SYSTEM] Stopping reader for camera ID: {cam_id}")
            active_readers[cam_id].stop()
            active_readers.pop(cam_id, None)
            active_camera_names.pop(cam_id, None)
            with frame_lock:
                latest_display_frames.pop(cam_id, None)

    # Start or update active readers
    for cam_id, name, url in active_cameras:
        sanitized_url = sanitize_rtsp_url(url)
        active_camera_names[cam_id] = name
        
        if cam_id in active_readers:
            if active_readers[cam_id].url != sanitized_url:
                active_readers[cam_id].change_url(sanitized_url)
        else:
            print(f"[SYSTEM] Starting reader for camera: {name} (ID: {cam_id})")
            reader = RTSPVideoReader(sanitized_url, cam_id=cam_id, name=name)
            reader.start()
            active_readers[cam_id] = reader

# YOLO Model Path
MODEL_PATH = "yolov8n_lpr_openvino_model"  # OpenVINO format for Intel UHD 630 GPU inference

class RTSPVideoReader:
    def __init__(self, url, cam_id=None, name=""):
        self.url = url
        self.cam_id = cam_id
        self.name = name
        self.cap = None
        self.q = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        self.offline_start_time = None
        self.alert_sent = False
        # Clean local proxy URL managed by go2rtc
        self.proxy_url = f"rtsp://127.0.0.1:8554/camera{cam_id}"

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def change_url(self, new_url):
        print(f"[SYSTEM] Changing RTSP URL to: {new_url}")
        self.url = new_url

    def _handle_offline_tick(self):
        if self.offline_start_time is None:
            self.offline_start_time = time.time()
        else:
            duration = time.time() - self.offline_start_time
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = 'offline_threshold_minutes'")
                row = cursor.fetchone()
                threshold_min = int(row[0]) if row else 5
                conn.close()
            except Exception:
                threshold_min = 5
            
            if duration >= threshold_min * 60 and not self.alert_sent:
                self.alert_sent = True
                alert_text = f"⚠️ [相機離線警報] ⚠️\n🎥 攝影機：{self.name} (ID: {self.cam_id})\n❌ 狀態：已失去連線超過 {threshold_min} 分鐘！\n請儘速檢查網路線路或攝影機電源。"
                print(f"[SYSTEM] {alert_text}")
                send_telegram_text_async(alert_text)

    def _reader(self):
        while self.running:
            print(f"Connecting to RTSP proxy for camera {self.name}...")
            current_url = self.url
            open_url = self.proxy_url
            
            import os
            # Limit FFmpeg decoder to 1 thread per camera stream (down from 2).
            # 1 thread is sufficient for 1080p H.264 at 10-15fps; frees 1 core per camera.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|threads;1"
            with video_capture_lock:
                self.cap = cv2.VideoCapture(open_url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                print("Failed to open proxy stream, retrying...")
                self._handle_offline_tick()
                for _ in range(50):
                    if not self.running or self.url != current_url:
                        break
                    time.sleep(0.1)
                continue
 
            print("RTSP proxy connection established.")
            if self.alert_sent:
                recovery_text = f"✅ [相機連線恢復] ✅\n🎥 攝影機：{self.name} (ID: {self.cam_id})\n🟢 狀態：已恢復連線，系統監控運作中。"
                print(f"[SYSTEM] {recovery_text}")
                send_telegram_text_async(recovery_text)
            self.offline_start_time = None
            self.alert_sent = False

            while self.running:
                if self.url != current_url:
                    print("[SYSTEM] RTSP URL changed. Reconnecting...")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Stream read failure. Attempting reconnection...")
                    self._handle_offline_tick()
                    break
                
                # CPU OpenVINO Mode: Skip every other frame to target ~6fps effective processing.
                # 80ms sleep reduces raw decode rate to ~12fps per camera.
                # This keeps CPU usage low while remaining fast enough to catch motorcycles.
                if not hasattr(self, '_skip_counter'):
                    self._skip_counter = 0
                self._skip_counter += 1
                time.sleep(0.08)  # Limit decode to ~12 reads/sec per camera
                if self._skip_counter % 2 == 0:
                    continue  # Process 1 in 2 frames (effective ~6fps)
                
                # Keep only the latest frame in the queue
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            
            self.cap.release()
            # Sleep in 100ms chunks before reconnecting
            for _ in range(20):
                if not self.running or self.url != current_url:
                    break
                time.sleep(0.1)
            for _ in range(20):
                if not self.running or self.url != current_url:
                    break
                time.sleep(0.1)

    def get_frame(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

# Clean up recognized plate text to standard format (Alphanumeric uppercase, no hyphens)
def clean_plate_text(text):
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    # ── Improvement 2: Enhanced position-based character correction ──────────
    # Taiwan license plate formats and their split points:
    #   5-char: 2L + 3D  (e.g. AB-123)   → split=2
    #   6-char: 3L + 3D  (e.g. ABC-123)  → split=3
    #   7-char: 3L + 4D  (e.g. ABC-1234) → split=3
    #   8-char: 4L + 4D  (e.g. ABCD-1234)→ split=4  (electric scooter)
    # In the letter-zone, digits that look like letters are corrected to letters.
    # In the digit-zone, letters that look like digits are corrected to digits.
    DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    LETTER_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'J': '1',
                       'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'Y': '7', 'A': '4'}

    FORMAT_SPLITS = {5: 2, 6: 3, 7: 3, 8: 4}
    if len(cleaned) in FORMAT_SPLITS:
        split = FORMAT_SPLITS[len(cleaned)]
        prefix = cleaned[:split]
        suffix = cleaned[split:]

        # Only apply if the prefix 'looks like' a letter zone and suffix 'looks like' a digit zone
        letter_zone_score = sum(c.isalpha() or c in DIGIT_TO_LETTER for c in prefix)
        digit_zone_score  = sum(c.isdigit() or c in LETTER_TO_DIGIT for c in suffix)

        if letter_zone_score >= split - 1 and digit_zone_score >= len(suffix) - 1:
            corrected = [DIGIT_TO_LETTER.get(c, c) for c in prefix] + \
                        [LETTER_TO_DIGIT.get(c, c) for c in suffix]
            cleaned = "".join(corrected)

    return cleaned

# Calculate Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Coordinates of intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
        
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

# Global System Status Switch (True = Running, False = Paused)
system_enabled = True
last_gate_state = 'open'
last_left_gate_closed = True
last_right_gate_closed = True
manual_override = None  # Can be 'running', 'paused', or None
active_readers = {}       # Key: camera_id (int), Value: RTSPVideoReader instance
active_camera_names = {}  # Key: camera_id (int), Value: camera_name (str)

# Global variables for gate control state machine overrides (Motion and Plate tracking)
last_motion_time = time.time()
last_plate_time = 0.0

# Check if the gate is closed based on edge density and intensity variance in the left/right halves of the gate ROI
def check_gate_closed(frame):
    if frame is None:
        return True, True, True
    
    try:
        # Check if it is monitoring start hours (05:00 to 22:00)
        # 早上 05:00 至晚上 22:00 自動啟動監控（大門強制判定為開啟 False，以防白天光影誤判）
        # 晚上 22:00 至隔天早上 05:00 自動關閉啟動（啟用實際大門關閉偵測，若關門則暫停監控）
        import datetime
        now = datetime.datetime.now()
        current_time = now.time()
        day_start = datetime.time(5, 0)
        day_end = datetime.time(22, 0)
        if day_start <= current_time <= day_end:
            return False, False, False

        # Crop the gate ROI (1920x1080 space)
        # Gate starts at y=0, goes to y=75, x from 600 to 1250 (total width 650)
        # We split it into left half (x: 600 to 925) and right half (x: 925 to 1250)
        roi_left = frame[0:75, 600:925]
        roi_right = frame[0:75, 925:1250]
        
        if roi_left.size == 0 or roi_right.size == 0:
            return True, True, True
            
        def get_half_score(roi_half):
            gray = cv2.cvtColor(roi_half, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobelx = np.absolute(sobelx)
            density = np.mean(abs_sobelx > 20) * 100
            std_val = np.std(gray)
            return std_val + density * 2.0

        score_left = get_half_score(roi_left)
        score_right = get_half_score(roi_right)
        
        left_closed = score_left > 22.5
        right_closed = score_right > 22.5
        
        # The gate is considered closed ONLY IF both halves are closed.
        # If either half is open, the gate is open more than half-way.
        is_closed = (left_closed and right_closed)
        return is_closed, left_closed, right_closed
    except Exception as e:
        print(f"Error checking gate status: {e}")
        return True, True, True

# Check if two plate strings are similar with character tolerance (maximum 2 character differences)
# Increased from 1 to 2 to handle OCR variants (e.g., ADB8937 vs TADB8937, ADB8957)
def is_similar_plate(p1, p2):
    if p1 == p2:
        return True
    # Length difference > 2 means totally different plate
    if abs(len(p1) - len(p2)) > 2:
        return False
    
    # Align to the longer string
    l1, l2 = list(p1), list(p2)
    while len(l1) < len(l2):
        l1.insert(0, '?')  # Pad from left (handles leading char misreads like TADB8937)
    while len(l2) < len(l1):
        l2.insert(0, '?')
        
    diff = 0
    for c1, c2 in zip(l1, l2):
        if c1 != c2:
            diff += 1
            if diff > 2:  # Allow up to 2 character differences
                return False
    return True

# Determine watchlist entry/exit status and suppression status (duplicate checks)
def determine_watchlist_action_and_check_suppression(cursor, plate_number):
    """
    Query the database for the last alert of this plate today.
    Returns (action, should_suppress).
    """
    cursor.execute("""
        SELECT action, timestamp, strftime('%s', 'now') - strftime('%s', timestamp) as elapsed_seconds
        FROM alerts
        WHERE plate_number = ? AND date(timestamp, 'localtime') = date('now', 'localtime')
        ORDER BY timestamp DESC LIMIT 1
    """, (plate_number,))
    row = cursor.fetchone()
    if not row:
        return 'ENTRY', False
    
    last_action, last_timestamp, elapsed_seconds = row
    if not last_action:
        last_action = 'ENTRY'
        
    if elapsed_seconds is None:
        return 'ENTRY', False
        
    if elapsed_seconds < 180:
        return last_action, True
    else:
        new_action = 'EXIT' if last_action == 'ENTRY' else 'ENTRY'
        return new_action, False

# Classify vehicle type based on Taiwan license plate coding format rules
def classify_vehicle_type(plate):
    p = plate.replace("-", "").replace(" ", "").upper()
    
    # Electric motorcycle starts with EM
    if p.startswith("EM"):
        return "MOTORCYCLE"
        
    letters_count = sum(c.isalpha() for c in p)
    digits_count = sum(c.isdigit() for c in p)
    
    # 6-character plates
    if len(p) == 6:
        # 3 letters + 3 digits or 3 digits + 3 letters -> Motorcycle
        if letters_count == 3 and digits_count == 3:
            return "MOTORCYCLE"
        else:
            return "CAR"
            
    # 7-character plates
    elif len(p) == 7:
        # 2 letters + 5 digits -> Motorcycle (micro electric scooter)
        if letters_count == 2 and digits_count == 5:
            return "MOTORCYCLE"
        
        # 3 letters + 4 digits
        if letters_count == 3 and digits_count == 4:
            prefix3 = p[:3]
            first_char = p[0]
            second_char = p[1]
            
            # 1. Electric vehicles (E...)
            if first_char == 'E':
                # Electric motorcycles: EMA, EMB, EMC, EMD...EZZ (second char is M to Z)
                if second_char >= 'M':
                    return "MOTORCYCLE"
                else:
                    return "CAR"
            
            # 2. L-prefix (Heavy & Ordinary Motorcycles)
            if first_char == 'L':
                return "MOTORCYCLE"
                
            # 3. J, M, N, P, Q, S, X, Y, Z (Motorcycles)
            if first_char in ['J', 'M', 'N', 'P', 'Q', 'S', 'X', 'Y', 'Z']:
                return "MOTORCYCLE"
                
            # 4. H-prefix (HEA-HZZ are motorcycles, others like HAA-HDZ are cars)
            if first_char == 'H':
                if second_char >= 'E':
                    return "MOTORCYCLE"
                else:
                    return "CAR"
                    
            # 5. K-prefix (KQE-KQZ, KUA-KZZ are motorcycles, others are cars/trucks)
            if first_char == 'K':
                if (prefix3 >= 'KQE' and prefix3 <= 'KQZ') or (prefix3 >= 'KUA' and prefix3 <= 'KZZ'):
                    return "MOTORCYCLE"
                else:
                    return "CAR"
                    
            # 6. W-prefix (WFA-WZZ are motorcycles, others are welfare cars)
            if first_char == 'W':
                if second_char >= 'F':
                    return "MOTORCYCLE"
                else:
                    return "CAR"
            
            # 7. Default for A, B, C, D, F, G, R, T, V
            if first_char in ['A', 'B', 'C', 'D', 'F', 'G', 'R', 'T', 'V']:
                return "CAR"
                
    # Default fallback
    return "CAR"

def send_telegram_notification_async(plate, category, description, camera_name, crop_filename, action='ENTRY'):
    def run():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'tg_enabled'")
            tg_enabled = cursor.fetchone()
            cursor.execute("SELECT value FROM settings WHERE key = 'tg_bot_token'")
            tg_token = cursor.fetchone()
            cursor.execute("SELECT value FROM settings WHERE key = 'tg_chat_id'")
            tg_chat_id = cursor.fetchone()
            conn.close()
            
            if not tg_enabled or tg_enabled[0] != '1':
                return
            if not tg_token or not tg_token[0] or not tg_chat_id or not tg_chat_id[0]:
                return
                
            token = tg_token[0].strip()
            chat_id = tg_chat_id[0].strip()
            
            action_zh = "進入" if action == 'ENTRY' else "離開"
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            caption = (
                f"🚨 追蹤車牌警報 ({action_zh})！\n"
                f"📍 偵測鏡頭：{camera_name}\n"
                f"🚗 車牌號碼：{plate} ({category})\n"
                f"📝 說明備註：{description or '無'}\n"
                f"⏰ 偵測時間：{now_str}"
            )
            
            photo_path = os.path.join(CROPS_DIR, crop_filename)
            if os.path.exists(photo_path):
                success, response = send_telegram_photo(token, chat_id, photo_path, caption)
                if not success:
                    print(f"[TELEGRAM] Failed to send alert photo: {response}")
                else:
                    print(f"[TELEGRAM] Alert push sent successfully for {plate}")
            else:
                success, response = send_telegram_text(token, chat_id, caption)
                if not success:
                    print(f"[TELEGRAM] Failed to send alert text: {response}")
                else:
                    print(f"[TELEGRAM] Alert text push sent successfully for {plate}")
        except Exception as e:
            print(f"[TELEGRAM] Error in sending thread: {e}")
            
    threading.Thread(target=run, daemon=True).start()

def send_telegram_photo(token, chat_id, photo_path, caption):
    import urllib.request
    import uuid
    try:
        boundary = f"Boundary-{uuid.uuid4().hex}"
        with open(photo_path, 'rb') as f:
            file_content = f.read()
        filename = os.path.basename(photo_path)
        
        parts = []
        parts.append(f"--{boundary}".encode('utf-8'))
        parts.append(f'Content-Disposition: form-data; name="chat_id"'.encode('utf-8'))
        parts.append(b'')
        parts.append(str(chat_id).encode('utf-8'))
        
        parts.append(f"--{boundary}".encode('utf-8'))
        parts.append(f'Content-Disposition: form-data; name="caption"'.encode('utf-8'))
        parts.append(b'')
        parts.append(caption.encode('utf-8'))
        
        parts.append(f"--{boundary}".encode('utf-8'))
        parts.append(f'Content-Disposition: form-data; name="photo"; filename="{filename}"'.encode('utf-8'))
        parts.append(f'Content-Type: image/jpeg'.encode('utf-8'))
        parts.append(b'')
        parts.append(file_content)
        
        parts.append(f"--{boundary}--".encode('utf-8'))
        
        body = b'\r\n'.join(parts)
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        
        req = urllib.request.Request(url, data=body)
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        req.add_header('Content-Length', str(len(body)))
        
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = response.read().decode('utf-8')
            return True, res_data
    except Exception as e:
        return False, str(e)

def send_telegram_text(token, chat_id, text):
    import urllib.request
    import json
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = response.read().decode('utf-8')
            return True, res_data
    except Exception as e:
        return False, str(e)

def send_telegram_text_async(text):
    def run():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings WHERE key IN ('tg_enabled', 'tg_bot_token', 'tg_chat_id')")
            rows = cursor.fetchall()
            conn.close()
            
            settings_dict = {r[0]: r[1] for r in rows}
            enabled = settings_dict.get('tg_enabled', '0') == '1'
            token = settings_dict.get('tg_bot_token', '').strip()
            chat_id = settings_dict.get('tg_chat_id', '').strip()
            
            if enabled and token and chat_id:
                success, response = send_telegram_text(token, chat_id, text)
                if not success:
                    print(f"[SYSTEM] Failed to send Telegram alert: {response}")
        except Exception as e:
            print(f"[SYSTEM] Error in send_telegram_text_async: {e}")
            
    t = threading.Thread(target=run, daemon=True)
    t.start()

# Thread Pool HTTP Server: replaces ThreadingMixIn (one-thread-per-request).
# Canvas polling sends 2 cameras × 10fps = 20 req/s. ThreadingMixIn would create
# 20 new threads/second. A pool of 8 workers handles all requests with zero thread
# creation overhead, cutting HTTP-related CPU from ~80% to ~20%.
from concurrent.futures import ThreadPoolExecutor

class ThreadedHTTPServer(TCPServer):
    """HTTP server using a fixed thread pool (max 8 workers) instead of one-thread-per-request."""
    allow_reuse_address = True
    _pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='http_worker')

    def process_request(self, request, client_address):
        self._pool.submit(self.process_request_thread, request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self):
        super().server_close()
        self._pool.shutdown(wait=False)

# Custom HTTP Request Handler
class LPRHTTPServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute console logs for HTTP requests to prevent cluttering the output
        pass

    def check_auth(self, auth_header):
        username = "admin"
        password = "admin123"
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'web_username'")
            row_u = cursor.fetchone()
            cursor.execute("SELECT value FROM settings WHERE key = 'web_password'")
            row_p = cursor.fetchone()
            conn.close()
            if row_u and row_u[0]:
                username = row_u[0]
            if row_p and row_p[0]:
                password = row_p[0]
        except Exception as e:
            print(f"[SYSTEM] Error reading auth settings: {e}")

        if not auth_header:
            return False
        if not auth_header.startswith('Basic '):
            return False
        
        import base64
        try:
            encoded_credentials = auth_header.split(' ', 1)[1]
            decoded_bytes = base64.b64decode(encoded_credentials)
            decoded_str = decoded_bytes.decode('utf-8')
            req_username, req_password = decoded_str.split(':', 1)
            return req_username == username and req_password == password
        except Exception:
            return False

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Check HTTP Basic Authentication
        auth_header = self.headers.get('Authorization')
        if not self.check_auth(auth_header):
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="LPR System"')
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b'<h1>401 Unauthorized</h1>')
            return

        global latest_display_frames
        
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # 0. System status check
        if path == "/api/system_status":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            status_str = "running" if system_enabled else "paused"
            gate_str = "closed" if last_gate_state == 'closed' else "open"
            self.wfile.write(json.dumps({
                "status": status_str, 
                "gate": gate_str, 
                "auto_control": manual_override is None
            }).encode('utf-8'))

        # GET /api/zones – return current zone polygons (1920×1080 coords)
        elif path == "/api/zones":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            with zone_config_lock:
                payload = {str(cam_id): poly.tolist()
                           for cam_id, poly in GATE_ZONE_POLYGONS.items()}
            self.wfile.write(json.dumps(payload).encode('utf-8'))

        # GET /api/snapshot?cam_id=X – latest camera frame as JPEG
        elif path == "/api/snapshot":
            cam_id_str = query.get('cam_id', ['1'])[0]
            try:
                cam_id_req = int(cam_id_str)
            except ValueError:
                cam_id_req = 1
            with frame_lock:
                frame_bytes = latest_display_frames.get(cam_id_req)
            if frame_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(404)
                self.send_cors_headers()
                self.end_headers()

        # Telegram settings check
        elif path == "/api/settings/telegram":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings WHERE key LIKE 'tg_%'")
            rows = cursor.fetchall()
            conn.close()
            
            settings_dict = {r[0]: r[1] for r in rows}
            self.wfile.write(json.dumps(settings_dict).encode('utf-8'))

        # 0A. Get Camera Settings API
        elif path == "/api/camera":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, rtsp_url FROM cameras WHERE is_active = 1")
            res = cursor.fetchone()
            if not res:
                cursor.execute("SELECT id, name, rtsp_url FROM cameras LIMIT 1")
                res = cursor.fetchone()
            conn.close()
            
            if res:
                payload = {"id": res[0], "name": res[1], "rtsp_url": res[2]}
            else:
                payload = {"id": 1, "name": "學校大門", "rtsp_url": RTSP_URL}
                
            self.wfile.write(json.dumps(payload).encode('utf-8'))

        # 0B. Get All Cameras List API
        elif path == "/api/cameras":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, rtsp_url, is_active FROM cameras ORDER BY id ASC")
            rows = cursor.fetchall()
            conn.close()
            
            payload = [{"id": r[0], "name": r[1], "rtsp_url": r[2], "is_active": r[3]} for r in rows]
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        elif path == "/api/stream":
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_cors_headers()
            self.end_headers()
            
            cam_id_str = query.get('id', [''])[0].strip()
            try:
                cam_id = int(cam_id_str)
            except ValueError:
                active_ids = list(active_readers.keys())
                cam_id = active_ids[0] if active_ids else None
                
            print(f"Client connected to live stream for camera ID: {cam_id}.")
            try:
                last_frame_bytes = None
                while True:
                    frame_bytes = None
                    if cam_id is not None:
                        with frame_lock:
                            frame_bytes = latest_display_frames.get(cam_id)
 
                    if frame_bytes:
                        last_frame_bytes = frame_bytes
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    elif last_frame_bytes:
                        # Re-send last known frame to keep the connection alive and avoid blank flicker
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(last_frame_bytes)}\r\n\r\n'.encode())
                        self.wfile.write(last_frame_bytes)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    
                    time.sleep(0.10)  # 10fps stream: consistent with 10fps display throttle in main loop
            except Exception as e:
                print(f"[SYSTEM] Stream client disconnected for camera ID: {cam_id}. Error: {e}")
                return

        # 1B. Single-frame JPEG endpoint for canvas polling
        # Canvas polling (/api/frame) replaces MJPEG (/api/stream) to eliminate
        # browser reconnect/white-flash. Each request returns latest JPEG independently.
        elif path == "/api/frame":
            cam_id_str = query.get('id', [''])[0].strip()
            try:
                cam_id = int(cam_id_str)
            except ValueError:
                active_ids = list(active_readers.keys())
                cam_id = active_ids[0] if active_ids else None

            with frame_lock:
                frame_bytes = latest_display_frames.get(cam_id) if cam_id is not None else None

            if frame_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(frame_bytes)))
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(204)  # No content yet
                self.send_cors_headers()
                self.end_headers()

        # 2. Get Recent Detections API (with Watchlist Join & Date Range & Pagination)
        elif path == "/api/detections":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()

            limit = int(query.get('limit', [50])[0])
            offset = int(query.get('offset', [0])[0])
            search = query.get('search', [''])[0].strip().upper()
            date_filter = query.get('date', [''])[0].strip()
            start_date = query.get('start_date', [''])[0].strip()
            end_date = query.get('end_date', [''])[0].strip()

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql_where = " WHERE 1=1"
            params = []

            camera_id_str = query.get('camera_id', [''])[0].strip()
            if camera_id_str:
                try:
                    sql_where += " AND d.camera_id = ?"
                    params.append(int(camera_id_str))
                except ValueError:
                    pass

            if search:
                sql_where += " AND d.plate_number LIKE ?"
                params.append(f"%{search}%")
            if start_date:
                sql_where += " AND date(d.timestamp, 'localtime') >= ?"
                params.append(start_date)
            if end_date:
                sql_where += " AND date(d.timestamp, 'localtime') <= ?"
                params.append(end_date)
            elif date_filter:
                sql_where += " AND date(d.timestamp, 'localtime') = ?"
                params.append(date_filter)

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM detections d" + sql_where, params)
            total_count = cursor.fetchone()[0]

            # Get paginated data
            sql_data = """
                SELECT d.id, datetime(d.timestamp, 'localtime') as local_time, 
                       d.plate_number, d.confidence, d.crop_image_path, d.full_image_path, d.vehicle_type,
                       w.category as watch_category, w.description as watch_description,
                       c.name as camera_name
                FROM detections d
                LEFT JOIN watchlist w ON d.plate_number = w.plate_number
                LEFT JOIN cameras c ON d.camera_id = c.id
            """ + sql_where + " ORDER BY d.timestamp DESC LIMIT ? OFFSET ?"
            
            data_params = list(params)
            data_params.extend([limit, offset])

            cursor.execute(sql_data, data_params)
            rows = cursor.fetchall()
            
            results_data = [dict(row) for row in rows]
            conn.close()

            response_payload = {
                "total": total_count,
                "data": results_data
            }
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        # 2A. Get Watchlist API (with Pagination)
        elif path == "/api/watchlist":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            try:
                limit = int(query.get('limit', [10])[0])
            except (ValueError, TypeError, IndexError):
                limit = 10

            try:
                offset = int(query.get('offset', [0])[0])
            except (ValueError, TypeError, IndexError):
                offset = 0
            
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM watchlist")
            total_count = cursor.fetchone()[0]
            
            # Get paginated data
            cursor.execute("""
                SELECT plate_number, category, description, datetime(created_at, 'localtime') as created_at 
                FROM watchlist 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            conn.close()
            
            response_payload = {
                "total": total_count,
                "data": results
            }
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        # 2B. Get Active Alerts API
        elif path == "/api/alerts":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            camera_id_str = query.get('camera_id', [''])[0].strip()
            camera_filter = ""
            params = []
            if camera_id_str:
                try:
                    camera_filter = " AND a.camera_id = ?"
                    params = [int(camera_id_str)]
                except ValueError:
                    pass

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT a.id, a.detection_id, a.plate_number, a.category, a.description, a.action, 
                       datetime(a.timestamp, 'localtime') as local_time, d.vehicle_type,
                       c.name as camera_name
                FROM alerts a
                LEFT JOIN detections d ON a.detection_id = d.id
                LEFT JOIN cameras c ON a.camera_id = c.id
                WHERE date(a.timestamp, 'localtime') = date('now', 'localtime'){camera_filter} 
                ORDER BY a.timestamp DESC
            """, params)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            conn.close()
            self.wfile.write(json.dumps(results).encode('utf-8'))

        # 3. Analytics & Stats API
        elif path == "/api/stats":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()

            camera_id_str = query.get('camera_id', [''])[0].strip()
            camera_filter = ""
            params = []
            if camera_id_str:
                try:
                    camera_filter = " AND camera_id = ?"
                    params = [int(camera_id_str)]
                except ValueError:
                    pass

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # A. Total, cars and motorcycles today
            cursor.execute(f"""
                SELECT 
                    COUNT(*),
                    SUM(CASE WHEN vehicle_type = 'CAR' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN vehicle_type = 'MOTORCYCLE' THEN 1 ELSE 0 END)
                FROM detections 
                WHERE date(timestamp, 'localtime') = date('now', 'localtime'){camera_filter}
            """, params)
            total_today, cars_today, motorcycles_today = cursor.fetchone()
            total_today = total_today or 0
            cars_today = cars_today or 0
            motorcycles_today = motorcycles_today or 0

            # B. Hourly distribution today (24 hours) for cars and motorcycles
            cursor.execute(f"""
                SELECT strftime('%H', timestamp, 'localtime') as hour, 
                       SUM(CASE WHEN vehicle_type = 'CAR' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN vehicle_type = 'MOTORCYCLE' THEN 1 ELSE 0 END)
                FROM detections 
                WHERE date(timestamp, 'localtime') = date('now', 'localtime'){camera_filter}
                GROUP BY hour
            """, params)
            hourly_raw_data = cursor.fetchall()
            hourly_cars = {r[0]: r[1] for r in hourly_raw_data}
            hourly_motorcycles = {r[0]: r[2] for r in hourly_raw_data}
            
            hourly_cars_today = [hourly_cars.get(f"{h:02d}", 0) for h in range(24)]
            hourly_motorcycles_today = [hourly_motorcycles.get(f"{h:02d}", 0) for h in range(24)]
            hourly_today = [hourly_cars_today[h] + hourly_motorcycles_today[h] for h in range(24)]

            # C. Daily distribution for last 7 days
            weekly_params = list(params)
            cursor.execute(f"""
                SELECT date(timestamp, 'localtime') as day, COUNT(*) 
                FROM detections 
                WHERE timestamp >= datetime('now', '-7 days', 'localtime'){camera_filter}
                GROUP BY day 
                ORDER BY day ASC
            """, weekly_params)
            weekly_raw = cursor.fetchall()
            weekly_stats = [{"date": r[0], "count": r[1]} for r in weekly_raw]

            conn.close()

            stats = {
                "total_today": total_today,
                "cars_today": cars_today,
                "motorcycles_today": motorcycles_today,
                "hourly_today": hourly_today,
                "hourly_cars_today": hourly_cars_today,
                "hourly_motorcycles_today": hourly_motorcycles_today,
                "weekly": weekly_stats
            }
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        elif path == "/api/settings/auth":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'web_username'")
            res = cursor.fetchone()
            conn.close()
            
            username = res[0] if res else "admin"
            self.wfile.write(json.dumps({"web_username": username}).encode('utf-8'))

        elif path == "/api/settings/maintenance":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings WHERE key IN ('retention_days', 'offline_threshold_minutes')")
            rows = cursor.fetchall()
            conn.close()
            
            settings_dict = {r[0]: r[1] for r in rows}
            retention_days = int(settings_dict.get('retention_days', 30))
            offline_threshold = int(settings_dict.get('offline_threshold_minutes', 5))
            
            self.wfile.write(json.dumps({
                "retention_days": retention_days,
                "offline_threshold_minutes": offline_threshold
            }).encode('utf-8'))

        # 4. Serve Cropped Images
        elif path.startswith("/crops/"):
            filename = os.path.basename(path)
            file_path = os.path.join(CROPS_DIR, filename)
            self.serve_file(file_path, "image/jpeg")

        # 5. Serve Full Frame Images
        elif path.startswith("/fulls/"):
            filename = os.path.basename(path)
            file_path = os.path.join(FULLS_DIR, filename)
            self.serve_file(file_path, "image/jpeg")

        # 6. Serve Frontend Static Files
        else:
            # Default to index.html
            local_path = path.strip("/")
            if not local_path:
                local_path = "index.html"
            
            file_path = os.path.join(PUBLIC_DIR, local_path)
            
            # Content Type Mapping
            mime_types = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".ico": "image/x-icon"
            }
            _, ext = os.path.splitext(file_path)
            content_type = mime_types.get(ext.lower(), "text/plain")
            
            self.serve_file(file_path, content_type)

    def serve_file(self, file_path, content_type):
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        try:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_cors_headers()
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"500 Internal Server Error: {str(e)}".encode())

    def do_POST(self):
        # Check HTTP Basic Authentication
        auth_header = self.headers.get('Authorization')
        if not self.check_auth(auth_header):
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="LPR System"')
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b'<h1>401 Unauthorized</h1>')
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == "/api/system_status":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                global system_enabled, manual_override
                data = json.loads(post_data)
                status = data.get('status', '').strip().lower()
                
                if status == 'paused':
                    system_enabled = False
                    manual_override = 'paused'
                elif status == 'running':
                    system_enabled = True
                    manual_override = 'running'
                else:
                    raise ValueError("Invalid status value. Use 'running' or 'paused'.")
                
                status_str = "running" if system_enabled else "paused"
                gate_str = "closed" if last_gate_state == 'closed' else "open"
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True, 
                    "status": status_str, 
                    "gate": gate_str,
                    "auto_control": manual_override is None
                }).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        # POST /api/zones – update zone polygon for one camera (hot-reload, no restart)
        elif path == "/api/zones":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                global GATE_ZONE_POLYGONS, gate_zones
                data = json.loads(post_data)
                cam_id_upd = int(data['cam_id'])
                polygon_upd = data['polygon']  # list of [x, y]
                if len(polygon_upd) < 3:
                    raise ValueError("Polygon must have at least 3 points")
                with zone_config_lock:
                    GATE_ZONE_POLYGONS[cam_id_upd] = np.array(polygon_upd)
                    rebuild_gate_zones()
                    # Persist all zones to file
                    zones_serial = {str(k): v.tolist() for k, v in GATE_ZONE_POLYGONS.items()}
                    save_zone_config(zones_serial)
                print(f"[ZONE] Camera {cam_id_upd} zone updated: {polygon_upd}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "cam_id": cam_id_upd}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/log":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                print(f"[BROWSER CONSOLE] {data.get('type', 'INFO')}: {data.get('message', '')}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/settings/telegram":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                enabled = data.get('tg_enabled', '0')
                token = data.get('tg_bot_token', '').strip()
                chat_id = data.get('tg_chat_id', '').strip()
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('tg_enabled', ?)", (enabled,))
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('tg_bot_token', ?)", (token,))
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('tg_chat_id', ?)", (chat_id,))
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/settings/telegram/test":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                token = data.get('tg_bot_token', '').strip()
                chat_id = data.get('tg_chat_id', '').strip()
                
                if not token or not chat_id:
                    raise ValueError("Token and Chat ID cannot be empty")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT crop_image_path FROM detections ORDER BY timestamp DESC LIMIT 1")
                res = cursor.fetchone()
                conn.close()
                
                caption = "🔔 AI LPR 監控系統 - Telegram 推播功能測試成功！"
                photo_sent = False
                res_msg = ""
                
                if res and res[0]:
                    photo_path = os.path.join(CROPS_DIR, res[0])
                    if os.path.exists(photo_path):
                        success, res_msg = send_telegram_photo(token, chat_id, photo_path, caption)
                        photo_sent = success
                        
                if not photo_sent:
                    success, res_msg = send_telegram_text(token, chat_id, caption)
                
                if success:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
                else:
                    raise Exception(res_msg)
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/camera":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                name = data['name'].strip()
                rtsp_url = sanitize_rtsp_url(data['rtsp_url'].strip())
                
                if not name or not rtsp_url:
                    raise ValueError("Name and RTSP URL cannot be empty")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cameras (id, name, rtsp_url) 
                    VALUES (1, ?, ?)
                """, (name, rtsp_url))
                conn.commit()
                conn.close()
                
                # Sync readers
                sync_active_readers()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/cameras":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                name = data['name'].strip()
                rtsp_url = sanitize_rtsp_url(data['rtsp_url'].strip())
                id_val = data.get('id')
                
                if not name or not rtsp_url:
                    raise ValueError("Name and RTSP URL cannot be empty")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                if id_val is not None:
                    # Update existing camera
                    cursor.execute("UPDATE cameras SET name = ?, rtsp_url = ? WHERE id = ?", (name, rtsp_url, id_val))
                else:
                    # Insert new camera
                    cursor.execute("INSERT INTO cameras (name, rtsp_url, is_active) VALUES (?, ?, 0)", (name, rtsp_url))
                
                conn.commit()
                conn.close()
                
                # Sync readers
                sync_active_readers()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/cameras/select":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                id_val = data['id']
                active_val = data.get('is_active')
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                if id_val == "all":
                    # Activate all cameras
                    cursor.execute("UPDATE cameras SET is_active = 1")
                elif id_val == "none":
                    # Deactivate all cameras
                    cursor.execute("UPDATE cameras SET is_active = 0")
                else:
                    id_int = int(id_val)
                    if active_val is not None:
                        # Set active state of single camera
                        cursor.execute("UPDATE cameras SET is_active = ? WHERE id = ?", (int(active_val), id_int))
                    else:
                        # Fallback: deactivate all others, activate selected camera
                        cursor.execute("UPDATE cameras SET is_active = 0")
                        cursor.execute("UPDATE cameras SET is_active = 1 WHERE id = ?", (id_int,))
                conn.commit()
                conn.close()
                
                # Sync active readers
                sync_active_readers()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        
        elif path == "/api/detections/edit":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                det_id = int(data['id'])
                new_plate = data['plate_number'].strip().upper()
                
                if not new_plate:
                    raise ValueError("Plate number cannot be empty")
                
                # 1. Classify vehicle type based on new plate
                new_v_type = classify_vehicle_type(new_plate)
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # 2. Update detections
                cursor.execute(
                    "UPDATE detections SET plate_number = ?, vehicle_type = ? WHERE id = ?",
                    (new_plate, new_v_type, det_id)
                )
                
                # 3. Synchronize alerts: delete old alerts for this detection ID
                cursor.execute("DELETE FROM alerts WHERE detection_id = ?", (det_id,))
                
                # 4. Check if corrected plate is on watchlist (VIP / Blacklist)
                cursor.execute("SELECT category, description FROM watchlist WHERE plate_number = ?", (new_plate,))
                watch_res = cursor.fetchone()
                
                if watch_res:
                    watch_category, watch_description = watch_res
                    # Fetch camera_id from the detection
                    cursor.execute("SELECT camera_id FROM detections WHERE id = ?", (det_id,))
                    cam_row = cursor.fetchone()
                    cam_id = cam_row[0] if cam_row else 1
                    
                    action_status, _ = determine_watchlist_action_and_check_suppression(cursor, new_plate)
                    cursor.execute(
                        "INSERT INTO alerts (detection_id, plate_number, category, description, camera_id, action) VALUES (?, ?, ?, ?, ?, ?)",
                        (det_id, new_plate, watch_category, watch_description, cam_id, action_status)
                    )
                
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        
        elif path == "/api/watchlist":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                plate_number = clean_plate_text(data['plate_number'].strip())
                category = data['category'].strip().upper()
                description = data.get('description', '').strip()
                
                if not plate_number or not category:
                    raise ValueError("Missing plate_number or category")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO watchlist (plate_number, category, description) 
                    VALUES (?, ?, ?)
                """, (plate_number, category, description))
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/settings/auth/save":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                username = data.get('web_username', 'admin').strip()
                password = data.get('web_password', '').strip()
                
                if not username:
                    raise ValueError("Username cannot be empty")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('web_username', ?)", (username,))
                if password:
                    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('web_password', ?)", (password,))
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/settings/maintenance/save":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                retention_days = int(data.get('retention_days', 30))
                offline_threshold = int(data.get('offline_threshold_minutes', 5))
                
                if retention_days < 1:
                    raise ValueError("Retention days must be at least 1")
                if offline_threshold < 1:
                    raise ValueError("Offline threshold minutes must be at least 1")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('retention_days', ?)", (str(retention_days),))
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('offline_threshold_minutes', ?)", (str(offline_threshold),))
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

    def do_DELETE(self):
        # Check HTTP Basic Authentication
        auth_header = self.headers.get('Authorization')
        if not self.check_auth(auth_header):
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="LPR System"')
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b'<h1>401 Unauthorized</h1>')
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        
        if path == "/api/watchlist":
            try:
                plate_number = clean_plate_text(query.get('plate_number', [''])[0].strip())
                if not plate_number:
                    raise ValueError("Missing plate_number parameter")
                    
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM watchlist WHERE plate_number = ?", (plate_number,))
                conn.commit()
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        elif path == "/api/cameras":
            try:
                id_val = int(query.get('id', [''])[0].strip())
                if not id_val:
                    raise ValueError("Missing id parameter")
                    
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute("SELECT is_active FROM cameras WHERE id = ?", (id_val,))
                row = cursor.fetchone()
                was_active = row and row[0] == 1
                
                cursor.execute("DELETE FROM cameras WHERE id = ?", (id_val,))
                
                if was_active:
                    cursor.execute("SELECT id, name, rtsp_url FROM cameras ORDER BY id ASC LIMIT 1")
                    fallback = cursor.fetchone()
                    if fallback:
                        fallback_id, fallback_name, fallback_url = fallback
                        cursor.execute("UPDATE cameras SET is_active = 1 WHERE id = ?", (fallback_id,))
                
                conn.commit()
                conn.close()
                
                # Sync active readers
                sync_active_readers()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

# Start the Web Server
def run_web_server():
    server_address = ('', 8081)
    httpd = ThreadedHTTPServer(server_address, LPRHTTPServerHandler)
    print("Web Dashboard Server running at http://localhost:8081")
    try:
        httpd.serve_forever()
    except Exception as e:
        print(f"Web server exception: {e}")

# Start background maintenance thread to clean up old database records and crop files
def start_maintenance_cleanup_thread():
    def cleanup_job():
        # Sleep 15 seconds on startup before running first cleanup to let the database settle
        time.sleep(15)
        while True:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = 'retention_days'")
                row = cursor.fetchone()
                retention_days = int(row[0]) if row else 30
                conn.close()
                
                # Calculate cutoff date
                cutoff_time = datetime.datetime.now() - datetime.timedelta(days=retention_days)
                cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"[MAINTENANCE] Running cleanup job. Retention days: {retention_days}. Cutoff time: {cutoff_str}")
                
                # Fetch records to delete files from disk
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT crop_image_path, full_image_path FROM detections WHERE timestamp < ?", (cutoff_str,))
                records_to_delete = cursor.fetchall()
                
                deleted_files_count = 0
                for crop_file, full_file in records_to_delete:
                    if crop_file:
                        p = os.path.join(CROPS_DIR, crop_file)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                                deleted_files_count += 1
                            except Exception:
                                pass
                    if full_file:
                        p = os.path.join(FULLS_DIR, full_file)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                                deleted_files_count += 1
                            except Exception:
                                pass
                
                # Delete alerts
                cursor.execute("""
                    DELETE FROM alerts WHERE detection_id IN (
                        SELECT id FROM detections WHERE timestamp < ?
                    )
                """, (cutoff_str,))
                deleted_alerts = cursor.rowcount
                
                # Delete detections
                cursor.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff_str,))
                deleted_detections = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                print(f"[MAINTENANCE] Cleanup finished. Deleted {deleted_files_count} files, {deleted_alerts} alerts, {deleted_detections} detections.")
            except Exception as e:
                print(f"[MAINTENANCE] Error in cleanup thread: {e}")
                
            # Sleep 12 hours
            time.sleep(12 * 3600)

    t = threading.Thread(target=cleanup_job, daemon=True)
    t.start()

# LPR Loop
def main():
    global last_motion_time, last_plate_time, system_enabled, last_gate_state, last_left_gate_closed, last_right_gate_closed, manual_override, active_readers, active_camera_names
    
    # Self-logging redirection
    import sys
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join(DATA_DIR, "engine.log")
    try:
        # Open in append mode, line-buffered
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
    except Exception as e:
        print(f"Failed to redirect logging to {log_path}: {e}")
        
    print(f"\n--- LPR ENGINE STARTING AT {datetime.datetime.now()} ---")

    # Clean up any running go2rtc/VLC instances at start
    try:
        subprocess.run("taskkill /f /im go2rtc.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("taskkill /f /im vlc.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
        
    init_db()
    print(f"Database initialized at {DB_PATH}")
    
    # Start Maintenance Cleanup Thread
    start_maintenance_cleanup_thread()

    # Load YOLO Model with OpenVINO on CPU
    # (Intel GPU compiler has bugs with certain layers, causing CISA routine errors and crashes)
    print(f"Loading YOLOv8 model from {MODEL_PATH} (device=cpu)...")
    model = YOLO(MODEL_PATH, task='detect')
    # Force first prediction to compile model on CPU
    import numpy as np
    _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model(_dummy, imgsz=640, device='cpu', verbose=False)
    print("[SYSTEM] YOLO model compiled on CPU successfully.")
    
    # OpenVINO CPU inference setup
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        for dev in devices:
            dev_name = core.get_property(dev, 'FULL_DEVICE_NAME')
            print(f"[SYSTEM] OpenVINO device: {dev} = {dev_name}")
        print("[SYSTEM] OpenVINO initialized on CPU successfully.")
    except Exception as e:
        print(f"[SYSTEM] OpenVINO setup info: {e}")
    # Limit PyTorch threads (for any residual torch ops in ultralytics pipeline)
    try:
        import torch
        torch.set_num_threads(2)
        print("[SYSTEM] PyTorch thread limit set to 2")
    except Exception as e:
        print(f"[SYSTEM] Could not limit PyTorch threads: {e}")
    
    # Load PaddleOCR
    print("Initializing PaddleOCR with enable_mkldnn=False...")
    ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=False, rec_batch_num=1)

    # Start all active RTSP readers from database
    sync_active_readers()

    # Start HTTP Web Server Thread
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()

    # Track active vehicles in the scene per camera
    tracked_vehicles = {}
    IOU_THRESHOLD = 0.60
    STATIONARY_TIMEOUT = 300.0  # Keep stationary/parked vehicles in track list for 5 minutes

    # Display frame throttle: limit MJPEG display updates to 10fps per camera
    # This prevents YOLO runs (200ms) from causing display gaps/stutter
    last_display_update = {}  # cam_id -> timestamp of last display frame update
    DISPLAY_UPDATE_INTERVAL = 0.10  # 10fps max display rate

    # Track recently logged plates per camera
    recently_logged_plates = {}
    LOGGED_SUPPRESSION_TIMEOUT = 90.0  # Suppress duplicate logs for the same plate within 1.5 minutes

    # Per-camera post-save cooldown: after ANY detection is saved, freeze saving for this camera
    # for SAVE_COOLDOWN_SECONDS to prevent the same vehicle being logged 3-5x during one pass.
    # Set to 6s (down from 20s) so motorcycles passing shortly after a car are still recorded.
    camera_save_cooldown = {}  # Key: cam_id, Value: timestamp of last save
    SAVE_COOLDOWN_SECONDS = 6.0  # 6 seconds per camera after any successful save

    print("AI LPR Engine started successfully. Monitoring stream...")
    
    # Frame counts per camera
    frame_counts = {}
    
    # Previous gray frames per camera
    prev_grays = {}
    
    # Gate closed confirm count
    closed_confirm_count = 0
    
    # LPR trigger frames per camera
    lpr_trigger_frames = {}

    # ── Improvement 1: Cross-frame best-plate candidate buffers ─────────────
    # During a 30-frame trigger window we accumulate every candidate plate
    # detection.  When the window closes (trigger_frames → 0) we pick the
    # single candidate with the highest combined YOLO + OCR confidence and
    # commit it to the database, instead of taking the first-passing result.
    #
    # Structure:  plate_candidates[cam_id] = [
    #   {'plate': str, 'yolo_conf': float, 'ocr_conf': float,
    #    'combined': float, 'cropped_plate': ndarray, 'full_frame': ndarray,
    #    'box': [x1,y1,x2,y2], 'timestamp_str': str},
    #   ...
    # ]
    plate_candidates = {}   # key: cam_id

    # ── Improvement 3: sv.PolygonZone gate detection zones ───────────────────
    # Loaded from zone_config.json (or DEFAULT_ZONE_POLYGONS if first run).
    # Updated live via POST /api/zones without restarting the engine.
    global GATE_ZONE_POLYGONS, gate_zones
    zone_data = load_zone_config()
    GATE_ZONE_POLYGONS = {int(k): np.array(v) for k, v in zone_data.items()}
    rebuild_gate_zones()


    # Last motion times per camera
    last_motion_times = {}

    # Last plate times per camera
    last_plate_times = {}

    # P-H: Consecutive motion counts per camera to filter out single-frame glitches
    consecutive_motion_counts = {}

    try:
        while True:
            # Check if "學校大門" is active. If not, default system_enabled to True under auto control
            has_gate_camera = False
            for active_name in list(active_camera_names.values()):
                if active_name == "學校大門":
                    has_gate_camera = True
                    break
            if not has_gate_camera and manual_override is None:
                system_enabled = True

            processed_any = False
            cam_ids = list(active_readers.keys())

            for cam_id in cam_ids:
                reader = active_readers.get(cam_id)
                if not reader:
                    continue
                frame = reader.get_frame()
                if frame is None:
                    continue

                processed_any = True
                cam_name = active_camera_names.get(cam_id, f"Camera {cam_id}")
                
                # Update frame count
                frame_counts[cam_id] = frame_counts.get(cam_id, 0) + 1
                cam_frame_count = frame_counts[cam_id]

                # Cheap motion detection check - run for EVERY frame
                # Use 320x180 diff map for better motorcycle detection (small vehicles)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (320, 180))
                
                has_motion = False
                driveway_motion = False
                prev_g = prev_grays.get(cam_id)
                if prev_g is not None:
                    diff = cv2.absdiff(prev_g, gray_small)
                    _, diff_thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
                    
                    # Threshold 150/120: filters out typical wind/leaves/compression noise (typically <80px), 
                    # while motorcycles (120-400px) and cars (300-2000px) will trigger reliably.
                    # We also check for consecutive motion frames to ignore single-frame keyframe glitches.
                    raw_motion_detected = False
                    if cam_name == "學校大門":
                        driveway_diff_raw = diff_thresh[14:170, 10:310]
                        changed_driveway_pixels = cv2.countNonZero(driveway_diff_raw)
                        if changed_driveway_pixels >= 150:  # lowered from 450 to catch motorcycles
                            raw_motion_detected = True
                    elif cam_name == "學校大門002":
                        driveway_diff_raw = diff_thresh[20:170, 10:310]
                        changed_driveway_pixels = cv2.countNonZero(driveway_diff_raw)
                        if changed_driveway_pixels >= 120:  # lowered from 350 to catch motorcycles
                            raw_motion_detected = True
                    else:
                        changed_pixels = cv2.countNonZero(diff_thresh)
                        if changed_pixels >= 80:
                            raw_motion_detected = True
                    
                    if raw_motion_detected:
                        consecutive_motion_counts[cam_id] = consecutive_motion_counts.get(cam_id, 0) + 1
                        # Require 2 consecutive frames of motion to confirm actual vehicle pass
                        if consecutive_motion_counts[cam_id] >= 2:
                            driveway_motion = True
                            has_motion = True
                    else:
                        consecutive_motion_counts[cam_id] = 0
                
                prev_grays[cam_id] = gray_small
                
                if driveway_motion:
                    last_motion_times[cam_id] = time.time()
                    # Minimum 8 seconds between trigger resets.
                    # 8s cooldown prevents pedestrian crowd from saturating YOLO.
                    last_trigger_time = camera_save_cooldown.get(f'trigger_{cam_id}', 0.0)
                    if lpr_trigger_frames.get(cam_id, 0) == 0 and (time.time() - last_trigger_time) >= 8.0:  # 8s cooldown
                        lpr_trigger_frames[cam_id] = 30  # 30 frames at ~6fps = 5 sec window
                        camera_save_cooldown[f'trigger_{cam_id}'] = time.time()
                elif has_motion:
                    last_motion_times[cam_id] = time.time()

                # Check gate state every 30 frames (approx. every 1 second) for "學校大門"
                if cam_name == "學校大門" and cam_frame_count % 30 == 0:
                    is_closed, left_closed, right_closed = check_gate_closed(frame)
                    
                    if is_closed:
                        closed_confirm_count += 1
                    else:
                        closed_confirm_count = 0
                    
                    current_gate_state = 'closed' if closed_confirm_count >= 2 else 'open'
                    
                    if current_gate_state == 'open':
                        last_left_gate_closed = False
                        last_right_gate_closed = False
                    else:
                        last_left_gate_closed = left_closed
                        last_right_gate_closed = right_closed
                    
                    if current_gate_state != last_gate_state:
                        print(f"[SYSTEM] Gate transitioned from {last_gate_state} to {current_gate_state}.")
                        manual_override = None  # Reset override on transition
                        last_gate_state = current_gate_state
                        if current_gate_state == 'open':
                            system_enabled = True
                            print("[SYSTEM] Gate opened. Automatic monitoring resumed.")
                        else:
                            system_enabled = False
                            print("[SYSTEM] Gate closed. Automatic monitoring paused.")
                    else:
                        if manual_override is None:
                            if current_gate_state == 'open':
                                system_enabled = True
                            else:
                                system_enabled = False

                # Throttle display updates to 10fps to prevent YOLO (200ms) from
                # causing display gaps. Only create display frame if enough time has passed.
                do_display_update = (time.time() - last_display_update.get(cam_id, 0)) >= DISPLAY_UPDATE_INTERVAL
                display_h, display_w = 360, 640  # P7: reduced from 960x540 to cut imencode CPU ~60%

                # Create display frame and draw gate indicator (only for 學校大門)
                display_frame = frame.copy()
                if cam_name == "學校大門":
                    left_color = (0, 0, 255) if last_left_gate_closed else (0, 255, 0)
                    right_color = (0, 0, 255) if last_right_gate_closed else (0, 255, 0)
                    
                    # Left half box
                    cv2.rectangle(display_frame, (600, 2), (925, 75), left_color, 2)
                    # Right half box
                    cv2.rectangle(display_frame, (925, 2), (1250, 75), right_color, 2)
                    
                    if last_gate_state == 'closed':
                        cv2.putText(display_frame, "GATE CLOSED", (610, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "GATE OPEN (>50%)", (610, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ── Always draw gate zone polygon on every display frame ────────────
                # Drawn here (before any 'continue') so it appears in ALL display
                # states: idle, motion, LPR trigger, and paused.
                zone_poly_draw = GATE_ZONE_POLYGONS.get(cam_id)
                if zone_poly_draw is not None:
                    cv2.polylines(display_frame, [zone_poly_draw], True, (0, 220, 255), 2)
                    cv2.putText(display_frame, "ZONE",
                                (int(zone_poly_draw[0][0]) + 6, int(zone_poly_draw[0][1]) + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

                # Check if LPR is paused for this specific camera
                camera_lpr_active = True
                if manual_override == 'paused':
                    camera_lpr_active = False
                elif cam_name == "學校大門":
                    camera_lpr_active = system_enabled

                if not camera_lpr_active:
                    cv2.putText(display_frame, "SYSTEM PAUSED (MONITORING INACTIVE)", (30, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if do_display_update:
                        display_small = cv2.resize(display_frame, (display_w, display_h))
                        _, img_encoded = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        with frame_lock:
                            latest_display_frames[cam_id] = img_encoded.tobytes()
                        last_display_update[cam_id] = time.time()
                    continue

                if not has_motion:
                    if do_display_update:
                        display_small = cv2.resize(display_frame, (display_w, display_h))
                        _, img_encoded = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        with frame_lock:
                            latest_display_frames[cam_id] = img_encoded.tobytes()
                        last_display_update[cam_id] = time.time()
                    continue
                
                # Run YOLO every 3 frames during trigger window.
                # 30-frame window ÷ 3 = 10 YOLO runs per trigger at ~6fps = covers 5 seconds.
                # A motorcycle passing in 1.5 seconds gets 3-4 YOLO attempts.
                run_lpr = (cam_frame_count % 3 == 0)
                cam_trigger_frames = lpr_trigger_frames.get(cam_id, 0)
                trigger_window_closing = False  # True on the very last frame of a trigger window
                if cam_trigger_frames > 0:
                    run_lpr = True
                    cam_trigger_frames -= 1
                    lpr_trigger_frames[cam_id] = cam_trigger_frames
                    if cam_trigger_frames == 24:
                        # First frame of window: reset candidate buffer for this camera
                        plate_candidates[cam_id] = []
                        print(f"[SYSTEM] {cam_name} driveway motion trigger: running LPR detection (30 frames).")
                    if cam_trigger_frames == 0:
                        trigger_window_closing = True  # Window just expired → commit best plate

                # Pre-YOLO display update: push current frame to stream NOW, before
                # the blocking model() call occupies CPU for ~200ms. This keeps the
                # MJPEG stream showing a fresh frame instead of a stale frozen one.
                if do_display_update:
                    display_small = cv2.resize(display_frame, (display_w, display_h))
                    _, img_encoded = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    with frame_lock:
                        latest_display_frames[cam_id] = img_encoded.tobytes()
                    last_display_update[cam_id] = time.time()

                if run_lpr:
                    # Mask out the timestamp watermark in the bottom-right corner to prevent false detections
                    detection_frame = frame.copy()
                    h, w, _ = detection_frame.shape
                    detection_frame[int(h*0.9):h, int(w*0.7):w] = 0
                    
                    # Crop to the active ROI area before passing to YOLO to speed up inference and ignore background pixels
                    crop_y1, crop_y2, crop_x1, crop_x2 = 0, h, 0, w
                    if cam_name == "學校大門":
                        crop_y1 = max(0, int(0.05 * h))
                        crop_y2 = min(h, int(0.95 * h))
                        crop_x1 = max(0, int(0.02 * w))
                        crop_x2 = min(w, int(0.98 * w))
                    elif cam_name == "學校大門002":
                        crop_y1 = max(0, int(0.05 * h))
                        crop_y2 = min(h, int(0.95 * h))
                        crop_x1 = max(0, int(0.02 * w))
                        crop_x2 = min(w, int(0.98 * w))
                    
                    yolo_input = detection_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    results = model(yolo_input, conf=0.15, imgsz=640, verbose=False)
                    current_time = time.time()
                    
                    # Check if any plates were found (result boxes are relative to cropped image)
                    any_plates_detected = False
                    for result in results:
                        if len(result.boxes) > 0:
                            any_plates_detected = True
                            break
                    if any_plates_detected:
                        last_plate_times[cam_id] = current_time
                    
                    if cam_id not in tracked_vehicles:
                        tracked_vehicles[cam_id] = []
                    if cam_id not in recently_logged_plates:
                        recently_logged_plates[cam_id] = {}

                    tracked_vehicles[cam_id] = [v for v in tracked_vehicles[cam_id] if current_time - v["last_seen"] < STATIONARY_TIMEOUT]
                    recently_logged_plates[cam_id] = {k: v for k, v in recently_logged_plates[cam_id].items() if current_time - v < LOGGED_SUPPRESSION_TIMEOUT}

                    # ── Build supervision Detections for zone filtering ────────
                    sv_detections_list = []
                    for result in results:
                        if len(result.boxes) > 0:
                            det = sv.Detections.from_ultralytics(result)
                            # Shift coordinates from cropped-frame space → full-frame space
                            det.xyxy[:, [0, 2]] += crop_x1
                            det.xyxy[:, [1, 3]] += crop_y1
                            sv_detections_list.append(det)

                    if sv_detections_list:
                        all_detections = sv.Detections.merge(sv_detections_list)
                    else:
                        all_detections = sv.Detections.empty()

                    # Apply PolygonZone filter: keep only detections inside gate zone
                    gate_zone = gate_zones.get(cam_id)
                    if gate_zone is not None and len(all_detections) > 0:
                        zone_mask = gate_zone.trigger(detections=all_detections)
                        filtered_detections = all_detections[zone_mask]
                        n_total   = len(all_detections)
                        n_in_zone = len(filtered_detections)
                        if n_total > n_in_zone:
                            print(f"[ZONE] {cam_name}: {n_total - n_in_zone}/{n_total} detections rejected (outside gate zone)")
                    else:
                        filtered_detections = all_detections

                    # ── Iterate over zone-filtered plate detections ───────────
                    for det_idx in range(len(filtered_detections)):
                        xyxy = filtered_detections.xyxy[det_idx]
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        # Keep the legacy top-left corner filter as backup
                        # (camera name overlay area – already outside zone polygon for most cameras)
                        if x1 < 400 and y1 < 80:
                            continue

                        confidence = float(filtered_detections.confidence[det_idx])

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        h, w, _ = frame.shape
                        pad_x = int((x2 - x1) * 0.15)
                        pad_y = int((y2 - y1) * 0.15)
                        px1 = max(0, x1 - pad_x)
                        py1 = max(0, y1 - pad_y)
                        px2 = min(w, x2 + pad_x)
                        py2 = min(h, y2 + pad_y)

                        cropped_plate = frame[py1:py2, px1:px2]
                        if cropped_plate.size == 0:
                            continue

                        crop_h, crop_w, _ = cropped_plate.shape
                        # Relaxed minimum plate size from 40x15 to 25x10 to capture small/distant motorcycle plates
                        if crop_w < 25 or crop_h < 10:
                            continue

                        cropped_plate_padded = cv2.copyMakeBorder(
                            cropped_plate,
                            top=10, bottom=10, left=15, right=15,
                            borderType=cv2.BORDER_REPLICATE
                        )

                        # Apply Image Enhancements for OCR accuracy:
                        # A. Resize to a uniform height of 64px (maintaining aspect ratio)
                        target_h = 64
                        ph, pw, _ = cropped_plate_padded.shape
                        target_w = int(pw * (target_h / ph))
                        enhanced = cv2.resize(cropped_plate_padded, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                        
                        # B. Local Contrast Enhancement using CLAHE on Luminance channel
                        try:
                            ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
                            y_channel, cr, cb = cv2.split(ycrcb)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            y_channel = clahe.apply(y_channel)
                            ycrcb = cv2.merge([y_channel, cr, cb])
                            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                        except Exception as e:
                            print(f"[SYSTEM] CLAHE enhancement failed: {e}")
                            
                        # C. Sharpening using Unsharp Mask (Gaussian Blur subtraction)
                        try:
                            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                            enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
                        except Exception as e:
                            print(f"[SYSTEM] Sharpening failed: {e}")

                        ocr_res = ocr.ocr(enhanced)
                        plate_text = ""
                        ocr_conf = 0.0
                        
                        if ocr_res and len(ocr_res) > 0:
                            res_dict = ocr_res[0]
                            texts = res_dict.get('rec_texts', [])
                            scores = res_dict.get('rec_scores', [])
                            plate_text = "".join(texts)
                            ocr_conf = max(scores) if scores else 0.0
                        cleaned_plate = clean_plate_text(plate_text)
                        
                        # Adaptive OCR confidence thresholds:
                        # - Plates matching TW motorcycle format (short, mixed) get a very low threshold (0.40)
                        # - Plates matching TW car format get a moderate threshold (0.50)
                        # - Other plates require high confidence (0.65) to prevent garbage
                        is_valid_format = False
                        is_motorcycle_format = False
                        if len(cleaned_plate) >= 4 and len(cleaned_plate) <= 10:
                            letters_count = sum(c.isalpha() for c in cleaned_plate)
                            digits_count = sum(c.isdigit() for c in cleaned_plate)
                            # Motorcycle: 3L+3D or 3D+3L (6-char) or 2L+4D / 3L+4D / special EM prefix (7-char)
                            if len(cleaned_plate) == 6 and letters_count == 3 and digits_count == 3:
                                is_motorcycle_format = True
                                is_valid_format = True
                            elif len(cleaned_plate) == 7 and letters_count == 3 and digits_count == 4:
                                is_motorcycle_format = True
                                is_valid_format = True
                            elif 5 <= len(cleaned_plate) <= 7 and letters_count >= 2 and digits_count >= 3:
                                is_valid_format = True
                            
                        if is_motorcycle_format:
                            min_ocr_conf = 0.40  # Very permissive for confirmed motorcycle plate format
                        elif is_valid_format:
                            min_ocr_conf = 0.50  # Permissive for valid TW plate format
                        else:
                            min_ocr_conf = 0.65  # High confidence required for unclear formats
                        
                        if len(cleaned_plate) >= 4 and len(cleaned_plate) <= 10:
                            if ocr_conf < min_ocr_conf:
                                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Detected plate {cleaned_plate} on {cam_name} but rejected due to low confidence {round(ocr_conf, 2)} (required >= {min_ocr_conf})")

                        # ── Improvement 1: accumulate candidate instead of immediate save ──
                        # Only plates passing the OCR confidence gate are eligible.
                        if len(cleaned_plate) >= 4 and len(cleaned_plate) <= 10 and ocr_conf >= min_ocr_conf:
                            cv2.putText(display_frame, cleaned_plate, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            combined_conf = round(confidence + ocr_conf, 4)  # YOLO conf + OCR conf
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                            if cam_id not in plate_candidates:
                                plate_candidates[cam_id] = []

                            plate_candidates[cam_id].append({
                                "plate":         cleaned_plate,
                                "yolo_conf":     confidence,
                                "ocr_conf":      ocr_conf,
                                "combined":      combined_conf,
                                "cropped_plate": cropped_plate.copy(),
                                "full_frame":    frame.copy(),
                                "box":           [x1, y1, x2, y2],
                                "timestamp_str": timestamp_str,
                            })
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CANDIDATE] {cam_name}: {cleaned_plate} "
                                  f"(YOLO={confidence:.2f} OCR={ocr_conf:.2f} combined={combined_conf:.2f}) "
                                  f"[{len(plate_candidates[cam_id])} buffered]")

                # ── Commit best plate when trigger window closes ──────────────
                if trigger_window_closing:
                    candidates = plate_candidates.get(cam_id, [])
                    if candidates:
                        # Pick the candidate with the highest combined YOLO+OCR confidence
                        best = max(candidates, key=lambda c: c["combined"])
                        cleaned_plate = best["plate"]
                        ocr_conf      = best["ocr_conf"]
                        confidence_b  = best["yolo_conf"]
                        cropped_plate = best["cropped_plate"]
                        frame_b       = best["full_frame"]
                        box_b         = best["box"]
                        timestamp_str = best["timestamp_str"]
                        x1, y1, x2, y2 = box_b
                        current_time  = time.time()

                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BEST-PLATE] {cam_name}: "
                              f"selected '{cleaned_plate}' from {len(candidates)} candidates "
                              f"(YOLO={confidence_b:.2f} OCR={ocr_conf:.2f} combined={best['combined']:.2f})")

                        # Reset candidate buffer
                        plate_candidates[cam_id] = []

                        # --- Duplicate / cooldown checks (same logic as before) ---
                        if cam_id not in tracked_vehicles:
                            tracked_vehicles[cam_id] = []
                        if cam_id not in recently_logged_plates:
                            recently_logged_plates[cam_id] = {}

                        tracked_vehicles[cam_id] = [
                            v for v in tracked_vehicles[cam_id]
                            if current_time - v["last_seen"] < STATIONARY_TIMEOUT
                        ]
                        recently_logged_plates[cam_id] = {
                            k: v for k, v in recently_logged_plates[cam_id].items()
                            if current_time - v < LOGGED_SUPPRESSION_TIMEOUT
                        }

                        is_duplicate = False
                        matched_vehicle = None
                        max_iou = 0.0
                        for v in tracked_vehicles[cam_id]:
                            iou = calculate_iou(box_b, v["box"])
                            if iou > IOU_THRESHOLD and iou > max_iou:
                                matched_vehicle = v
                                max_iou = iou

                        if matched_vehicle:
                            matched_vehicle["box"]       = box_b
                            matched_vehicle["last_seen"] = current_time
                            is_duplicate = True

                            # If best plate is better than the already-recorded one, update DB
                            if cleaned_plate != matched_vehicle["plate"] and ocr_conf > matched_vehicle.get("confidence", 0.0):
                                old_plate = matched_vehicle["plate"]
                                det_id    = matched_vehicle.get("detection_id")
                                new_v_type = classify_vehicle_type(cleaned_plate)
                                if det_id:
                                    try:
                                        db_conn = sqlite3.connect(DB_PATH)
                                        cursor  = db_conn.cursor()
                                        cursor.execute(
                                            "UPDATE detections SET plate_number = ?, confidence = ?, vehicle_type = ? WHERE id = ?",
                                            (cleaned_plate, round(ocr_conf, 2), new_v_type, det_id)
                                        )
                                        cursor.execute("DELETE FROM alerts WHERE detection_id = ?", (det_id,))
                                        cursor.execute("SELECT category, description FROM watchlist WHERE plate_number = ?", (cleaned_plate,))
                                        watch_res = cursor.fetchone()
                                        cursor.execute("SELECT COUNT(*) FROM watchlist WHERE plate_number = ?", (old_plate,))
                                        was_old_alerted = cursor.fetchone()[0] > 0
                                        if watch_res:
                                            watch_category, watch_description = watch_res
                                            action_status, should_suppress = determine_watchlist_action_and_check_suppression(cursor, cleaned_plate)
                                            if not should_suppress:
                                                cursor.execute(
                                                    "INSERT INTO alerts (detection_id, plate_number, category, description, camera_id, action) VALUES (?, ?, ?, ?, ?, ?)",
                                                    (det_id, cleaned_plate, watch_category, watch_description, cam_id, action_status)
                                                )
                                                if not was_old_alerted:
                                                    cursor.execute("SELECT crop_image_path FROM detections WHERE id = ?", (det_id,))
                                                    crop_res  = cursor.fetchone()
                                                    crop_file = crop_res[0] if crop_res else ""
                                                    if crop_file:
                                                        send_telegram_notification_async(cleaned_plate, watch_category, watch_description, cam_name, crop_file, action_status)
                                        db_conn.commit()
                                        db_conn.close()
                                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updated stationary plate on {cam_name}: {old_plate} -> {cleaned_plate} (Conf: {round(ocr_conf, 2)})")
                                    except Exception as e:
                                        print(f"Error updating stationary plate: {e}")
                                matched_vehicle["plate"]        = cleaned_plate
                                matched_vehicle["confidence"]   = ocr_conf
                                matched_vehicle["vehicle_type"] = new_v_type
                                recently_logged_plates[cam_id][cleaned_plate] = current_time

                        if not is_duplicate:
                            for logged_plate in recently_logged_plates[cam_id]:
                                if is_similar_plate(cleaned_plate, logged_plate):
                                    is_duplicate = True
                                    recently_logged_plates[cam_id][logged_plate] = current_time
                                    break

                        if is_duplicate:
                            pass  # Skip insert but don't block the rest of the loop
                        else:
                            # Per-camera cooldown check
                            last_save_time = camera_save_cooldown.get(cam_id, 0.0)
                            if current_time - last_save_time < SAVE_COOLDOWN_SECONDS:
                                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [COOLDOWN] {cam_name} cooldown active "
                                      f"({SAVE_COOLDOWN_SECONDS - (current_time - last_save_time):.1f}s remaining), skipping {cleaned_plate}")
                            else:
                                v_type = classify_vehicle_type(cleaned_plate)
                                recently_logged_plates[cam_id][cleaned_plate] = current_time

                                crop_filename = f"crop_{cleaned_plate}_{timestamp_str}.jpg"
                                full_filename = f"full_{cleaned_plate}_{timestamp_str}.jpg"
                                crop_path     = os.path.join(CROPS_DIR, crop_filename)
                                full_path     = os.path.join(FULLS_DIR, full_filename)

                                cv2.imwrite(crop_path, cropped_plate)
                                cv2.imwrite(full_path, frame_b)

                                db_conn = sqlite3.connect(DB_PATH)
                                cursor  = db_conn.cursor()
                                cursor.execute(
                                    "INSERT INTO detections (plate_number, confidence, crop_image_path, full_image_path, vehicle_type, camera_id) "
                                    "VALUES (?, ?, ?, ?, ?, ?)",
                                    (cleaned_plate, round(ocr_conf, 2), crop_filename, full_filename, v_type, cam_id)
                                )
                                detection_id = cursor.lastrowid

                                cursor.execute("SELECT category, description FROM watchlist WHERE plate_number = ?", (cleaned_plate,))
                                watch_res = cursor.fetchone()
                                if watch_res:
                                    watch_category, watch_description = watch_res
                                    action_status, should_suppress = determine_watchlist_action_and_check_suppression(cursor, cleaned_plate)
                                    if not should_suppress:
                                        cursor.execute(
                                            "INSERT INTO alerts (detection_id, plate_number, category, description, camera_id, action) VALUES (?, ?, ?, ?, ?, ?)",
                                            (detection_id, cleaned_plate, watch_category, watch_description, cam_id, action_status)
                                        )
                                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] !!! ALERT !!! Tracked Vehicle {cleaned_plate} "
                                              f"({watch_category}) [{action_status}] detected on {cam_name}: {watch_description}")
                                        send_telegram_notification_async(cleaned_plate, watch_category, watch_description, cam_name, crop_filename, action_status)
                                    else:
                                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Watchlist Vehicle {cleaned_plate} detected but suppressed (duplicate within 3 min).")

                                db_conn.commit()
                                db_conn.close()

                                tracked_vehicles[cam_id].append({
                                    "box":          box_b,
                                    "plate":        cleaned_plate,
                                    "last_seen":    current_time,
                                    "vehicle_type": v_type,
                                    "detection_id": detection_id,
                                    "confidence":   ocr_conf
                                })

                                camera_save_cooldown[cam_id] = current_time
                                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DETECTED on {cam_name}: "
                                      f"{cleaned_plate} (YOLO={confidence_b:.2f} OCR={ocr_conf:.2f}) [best of {len(candidates)} candidates]")
                    else:
                        # Trigger window closed but no valid candidates were accumulated
                        plate_candidates[cam_id] = []
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BEST-PLATE] {cam_name}: trigger window closed with 0 valid candidates.")

                # Post-YOLO display update: re-evaluate throttle condition.
                # Since YOLO took ~200ms >> 100ms throttle interval, this always fires,
                # ensuring bounding boxes drawn above appear in the stream.
                if (time.time() - last_display_update.get(cam_id, 0)) >= DISPLAY_UPDATE_INTERVAL:
                    display_small = cv2.resize(display_frame, (display_w, display_h))
                    _, img_encoded = cv2.imencode('.jpg', display_small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    with frame_lock:
                        latest_display_frames[cam_id] = img_encoded.tobytes()
                    last_display_update[cam_id] = time.time()

            if not processed_any:
                time.sleep(0.03)
            else:
                time.sleep(0.02)  # ~50fps loop cap; actual per-camera rate limited by RTSP

    except KeyboardInterrupt:
        print("LPR Engine stopping...")
    finally:
        for cam_id, reader in list(active_readers.items()):
            reader.stop()
        stop_go2rtc()

if __name__ == "__main__":
    main()
