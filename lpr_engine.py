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
import base64
import urllib.parse
import urllib.request
import subprocess
from collections import defaultdict, Counter, deque  # moved from hot-path inline imports
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import TCPServer
from ultralytics import YOLO
from paddleocr import PaddleOCR
import supervision as sv  # Roboflow Supervision: PolygonZone gate filtering

# Configure Paths
PUBLIC_DIR = r"D:\AntiGravity\ai camera-gate\public"
DATA_DIR = r"D:\AntiGravity\lpr_data"
DB_PATH = os.path.join(DATA_DIR, "lpr.db")
CROPS_DIR   = os.path.join(DATA_DIR, "crops")
FULLS_DIR   = os.path.join(DATA_DIR, "fulls")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
ZONE_CONFIG_PATH = os.path.join(DATA_DIR, "zone_config.json")

# Ensure directories exist
os.makedirs(CROPS_DIR,   exist_ok=True)
os.makedirs(FULLS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR,  exist_ok=True)

# Global variables for sharing frames between threads
latest_display_frames = {} # Key: camera_id (int), Value: JPEG bytes (compressed frame)
frame_lock = threading.Lock()
video_capture_lock = threading.Lock()
readers_lock = threading.Lock()  # Guards active_readers dict across threads
# 改進 11: Module-level persistent DB connection (initialized in main()).
# Declared here so sync_active_readers() (module-level) can reuse it without
# opening a new sqlite3 connection on every call (~every 30 s).
db_conn_persistent = None
db_write_lock = threading.Lock()

# ── 改進 12: Auth credential cache ────────────────────────────────────────────
# check_auth() is called on every HTTP request (20/s from canvas polling).
# Caching avoids opening a new sqlite3 connection on each call.
# reload_auth_cache() is called once at startup and whenever credentials change.
_auth_cache: dict = {"username": "admin", "password": "3762828", "loaded": False}
_auth_cache_lock = threading.Lock()

def reload_auth_cache():
    """Load web credentials from DB into the in-memory cache."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings WHERE key IN ('web_username', 'web_password')")
        rows = cursor.fetchall()
        conn.close()
        d = {r[0]: r[1] for r in rows}
        with _auth_cache_lock:
            _auth_cache["username"] = d.get("web_username", "admin") or "admin"
            _auth_cache["password"] = d.get("web_password", "3762828") or "3762828"
            _auth_cache["loaded"] = True
    except Exception as e:
        print(f"[SYSTEM] reload_auth_cache error: {e}")

# ── 改進 14: Module-level display helpers ─────────────────────────────────────
# Previously defined as nested `def` inside the per-camera loop, so Python
# created new function objects + closures on every frame (18/s for 3 cameras).
# Extracting to module level means the function objects are built exactly ONCE.
# Variables that were captured by closure are now explicit parameters.
DISPLAY_W = 640  # MJPEG output width  (same value as before, now a named constant)
DISPLAY_H = 360  # MJPEG output height (same value as before)

def _make_display_frame(frame, cam_name, cam_id):
    """Create a display copy of frame with gate indicator and zone polygon overlay."""
    df = frame.copy()
    if cam_name == "學校大門":
        left_color  = (0, 0, 255) if last_left_gate_closed  else (0, 255, 0)
        right_color = (0, 0, 255) if last_right_gate_closed else (0, 255, 0)
        cv2.rectangle(df, (600, 2),  (925, 75),  left_color,  2)
        cv2.rectangle(df, (925, 2),  (1250, 75), right_color, 2)
        if last_gate_state == 'closed':
            cv2.putText(df, "GATE CLOSED",     (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(df, "GATE OPEN (>50%)", (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    _zpoly = GATE_ZONE_POLYGONS.get(cam_id)
    if _zpoly is not None:
        cv2.polylines(df, [_zpoly], True, (0, 220, 255), 2)
        cv2.putText(df, "ZONE",
                    (int(_zpoly[0][0]) + 6, int(_zpoly[0][1]) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return df

def _encode_and_push(df, cam_id, last_display_update):
    """Resize + JPEG-encode df and push to MJPEG buffer for cam_id."""
    ds = cv2.resize(df, (DISPLAY_W, DISPLAY_H))
    _, enc = cv2.imencode('.jpg', ds, [cv2.IMWRITE_JPEG_QUALITY, 65])
    with frame_lock:
        latest_display_frames[cam_id] = enc.tobytes()
    last_display_update[cam_id] = time.time()

# ── 光照自適應 YOLO 信心度 ────────────────────────────────────────────────────
# 每支攝影機獨立追蹤 Zone 區域內的亮度，並依此動態調整 YOLO conf 閾值。
# 解決問題：下午強光/逆光/陰天/夜間等不同光照條件下，固定 conf 值會造成
#   - conf 過高 → 漏報（下午逆光時車牌信心度降至 0.15-0.24，被過濾）
#   - conf 過低 → 誤報（良好光照下低信心雜訊框也會進入 OCR）
# 量測成本極低：每幀只對 320×180 縮小圖的 Zone mask 像素做 mean/std。
_cam_brightness_history: dict = {}   # cam_id -> deque(maxlen=60)  ~10s @ 6fps
_cam_yolo_conf: dict = {}            # cam_id -> 當前 adaptive conf 值
_cam_lighting_condition: dict = {}   # cam_id -> 當前光照狀態標籤 (str)
_cam_last_lighting_log: dict = {}    # cam_id -> 上次寫入 log 的 timestamp

def _compute_yolo_conf(mean_b: float, std_b: float):
    """依 Zone 區域平均亮度與標準差，計算適合的 YOLO conf 與狀態標籤。

    光照狀態對照表：
      overexposed : mean > 190 且 std < 38  → 強光/逆光/反射炫光，車牌曝白
      bright      : mean > 160              → 晴天良好光照
      normal      : mean > 95               → 一般日光
      dim         : mean > 45               → 黃昏/陰天/多雲
      dark        : mean ≤ 45               → 夜間/極暗

    Returns:
        (conf: float, condition: str)
    """
    if mean_b > 190 and std_b < 38:
        return 0.12, "overexposed"   # 炫光/曝白 → 降低閾值增加偵測率
    elif mean_b > 160:
        return 0.20, "bright"        # 晴天 → 提高閾值減少雜訊框
    elif mean_b > 95:
        return 0.15, "normal"        # 一般日光 → 標準閾值
    elif mean_b > 45:
        return 0.12, "dim"           # 黃昏/陰天 → 降低閾值
    else:
        return 0.10, "dark"          # 夜間 → 最低閾值

def vote_plate_text(ocr_history: list) -> str:
    """字符級 OCR 投票：對同一輛車的多幀讀取結果逐字符投票，產生最終車牌字串。

    原理：
      - ocr_history 是 [(plate_text, ocr_conf), ...] 的列表，來自同一 tracker_id 的
        所有幀讀取（輕量字串，不含 frame/crop）。
      - 先決定「主流長度」（出現最多的車牌長度），過濾掉離群長度讀取。
      - 對每個字符位置，用 ocr_conf 作為加權票，取加權票數最多的字符。
      - 最終結果比單純取最高 conf 更能抵抗偶發性誤讀（例如 0→O, 1→I）。

    Example:
      history = [("ABC123",0.72),("ABC123",0.68),("AB0123",0.65),("ABC123",0.70)]
      → positions: 0=A,1=B,2=C,3=1,4=2,5=3 → "ABC123"  (1 一票 0.65 vs 三票 1 sum=2.1)

    Args:
        ocr_history: list of (plate_str, conf_float) tuples

    Returns:
        Voted plate string, or '' if history is empty
    """
    if not ocr_history:
        return ''
    if len(ocr_history) == 1:
        return ocr_history[0][0]

    # 1. 決定主流長度（加權：conf 較高的讀取優先）
    length_weights: dict = {}
    for text, conf in ocr_history:
        ln = len(text)
        length_weights[ln] = length_weights.get(ln, 0.0) + conf
    majority_len = max(length_weights, key=length_weights.get)

    # 2. 過濾只保留主流長度的讀取
    filtered = [(t, c) for t, c in ocr_history if len(t) == majority_len]
    if not filtered:
        filtered = ocr_history  # fallback：保留全部

    # 3. 逐字符加權投票
    result_chars = []
    for pos in range(majority_len):
        char_votes: dict = {}
        for text, conf in filtered:
            if pos < len(text):
                ch = text[pos]
                char_votes[ch] = char_votes.get(ch, 0.0) + conf
        if char_votes:
            result_chars.append(max(char_votes, key=char_votes.get))
    return ''.join(result_chars)

def compute_direction(y_history: list) -> str:
    """根據車輛中心點 Y 座標歷史判斷行進方向。

    攝影機坐標系：Y=0 在畫面頂部，向下遞增。
      - 若後半段 Y 均值 > 前半段 → 車輛向下移動 → 進場 (ENTER)
      - 若後半段 Y 均值 < 前半段 → 車輛向上移動 → 離場 (EXIT)
      - 樣本不足或移動量不顯著 → UNKNOWN

    Args:
        y_history: list of int, center_y pixel coordinates per frame

    Returns:
        'ENTER', 'EXIT', or 'UNKNOWN'
    """
    if len(y_history) < 4:
        return 'UNKNOWN'
    half = len(y_history) // 2
    first_mean  = sum(y_history[:half]) / half
    second_mean = sum(y_history[half:]) / (len(y_history) - half)
    dy = second_mean - first_mean
    if dy > 8:    # 向下移動（Y 增加）→ 進場
        return 'ENTER'
    elif dy < -8:  # 向上移動（Y 減少）→ 離場
        return 'EXIT'
    return 'UNKNOWN'

# ── 每日統計報表系統 ───────────────────────────────────────────────────────────

def _register_cjk_font():
    """嘗試在 reportlab 中登錄繁體中文字型，回傳可用字型名稱。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        r"C:\Windows\Fonts\msjh.ttc",    # Microsoft JhengHei（微軟正黑）
        r"C:\Windows\Fonts\kaiu.ttf",     # 標楷體
        r"C:\Windows\Fonts\mingliu.ttc",  # 細明體
        r"C:\Windows\Fonts\meiryo.ttc",   # 日文（備用）
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('CJK', fp))
                return 'CJK'
            except Exception:
                pass
    return 'Helvetica'  # fallback：英文 only

def generate_daily_report_pdf(date_str: str) -> str:
    """生成指定日期的每日統計報表 PDF，儲存至 REPORTS_DIR。

    Args:
        date_str: 'YYYY-MM-DD' 格式

    Returns:
        生成的 PDF 絕對路徑
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    import datetime as _dt

    cn_font = _register_cjk_font()

    # ── 查詢資料庫 ────────────────────────────────────────────────────────────
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN direction='ENTER'      THEN 1 ELSE 0 END),
               SUM(CASE WHEN direction='EXIT'       THEN 1 ELSE 0 END),
               SUM(CASE WHEN vehicle_type='MOTORCYCLE' THEN 1 ELSE 0 END),
               SUM(CASE WHEN vehicle_type='CAR'    THEN 1 ELSE 0 END)
        FROM detections WHERE date(timestamp,'localtime') = ?
    """, (date_str,))
    r       = cursor.fetchone()
    total   = r[0] or 0; enters = r[1] or 0; exits = r[2] or 0
    motos   = r[3] or 0; cars   = r[4] or 0

    cursor.execute("""
        SELECT COUNT(*) FROM alerts a
        JOIN detections d ON a.detection_id = d.id
        WHERE date(d.timestamp,'localtime') = ?
    """, (date_str,))
    alerts_cnt = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT c.name, COUNT(*),
               SUM(CASE WHEN d.direction='ENTER' THEN 1 ELSE 0 END),
               SUM(CASE WHEN d.direction='EXIT'  THEN 1 ELSE 0 END)
        FROM detections d
        LEFT JOIN cameras c ON d.camera_id = c.id
        WHERE date(d.timestamp,'localtime') = ?
        GROUP BY d.camera_id, c.name ORDER BY COUNT(*) DESC
    """, (date_str,))
    cam_rows = cursor.fetchall()

    cursor.execute("""
        SELECT strftime('%H', timestamp,'localtime') hr, COUNT(*)
        FROM detections WHERE date(timestamp,'localtime') = ?
        GROUP BY hr ORDER BY hr
    """, (date_str,))
    hourly = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT plate_number, COUNT(*) cnt,
               MAX(confidence),
               MAX(CASE WHEN direction='ENTER' THEN 1 ELSE 0 END),
               MAX(CASE WHEN direction='EXIT'  THEN 1 ELSE 0 END)
        FROM detections WHERE date(timestamp,'localtime') = ?
        GROUP BY plate_number ORDER BY cnt DESC LIMIT 15
    """, (date_str,))
    top_plates = cursor.fetchall()
    conn.close()

    # ── 建立 PDF ─────────────────────────────────────────────────────────────
    pdf_path = os.path.join(REPORTS_DIR, f"daily_report_{date_str}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)

    def _sty(name, **kw):
        return ParagraphStyle(name, fontName=cn_font, **kw)

    title_sty    = _sty('T', fontSize=18, alignment=1, spaceAfter=4,
                         textColor=colors.HexColor('#1e3a5f'))
    subtitle_sty = _sty('S', fontSize=11, alignment=1, spaceAfter=10,
                         textColor=colors.HexColor('#555555'))
    head_sty     = _sty('H', fontSize=13, spaceBefore=12, spaceAfter=6,
                         textColor=colors.HexColor('#1e3a5f'))
    norm_sty     = _sty('N', fontSize=9,  spaceAfter=3,
                         textColor=colors.HexColor('#333333'))
    foot_sty     = _sty('F', fontSize=8,  alignment=1,
                         textColor=colors.grey)

    def _tbl(data, cols, hdr_color):
        tbl = Table(data, colWidths=cols)
        tbl.setStyle(TableStyle([
            ('FONTNAME',      (0,0), (-1,-1), cn_font),
            ('FONTSIZE',      (0,0), (-1,-1), 10),
            ('BACKGROUND',    (0,0), (-1, 0), colors.HexColor(hdr_color)),
            ('TEXTCOLOR',     (0,0), (-1, 0), colors.white),
            ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS',(0,1), (-1,-1),
             [colors.HexColor('#f4f6f9'), colors.white]),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        return tbl

    story = []
    now_str = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')

    story.append(Paragraph('學校大門 AI 車輛管制系統', title_sty))
    story.append(Paragraph(f'每日統計報表 — {date_str}', subtitle_sty))
    story.append(Paragraph(f'產生時間：{now_str}', norm_sty))
    story.append(HRFlowable(width='100%', thickness=2,
                             color=colors.HexColor('#1e3a5f'), spaceAfter=10))

    # 摘要
    story.append(Paragraph('■ 當日摘要', head_sty))
    story.append(_tbl([
        ['項目', '數值'],
        ['偵測總次數', str(total)],
        ['進場 (ENTER)', str(enters)],
        ['離場 (EXIT)', str(exits)],
        ['機車', str(motos)],
        ['汽車', str(cars)],
        ['追蹤名單命中', str(alerts_cnt)],
    ], [10*cm, 6*cm], '#1e3a5f'))
    story.append(Spacer(1, 0.4*cm))

    # 各攝影機
    if cam_rows:
        story.append(Paragraph('■ 各攝影機統計', head_sty))
        story.append(_tbl(
            [['攝影機', '偵測次數', '進場', '離場']] +
            [[r[0] or '未知', str(r[1] or 0), str(r[2] or 0), str(r[3] or 0)]
             for r in cam_rows],
            [8*cm, 4*cm, 3*cm, 3*cm], '#2d6a9f'))
        story.append(Spacer(1, 0.4*cm))

    # 各時段分佈
    story.append(Paragraph('■ 各時段偵測分佈（每3小時）', head_sty))
    hr_data = [['時段', '次數', '時段', '次數', '時段', '次數']]
    for i in range(0, 24, 3):
        row = []
        for j in range(3):
            h = f"{i+j:02d}" if i+j < 24 else ''
            row += [f"{h}:00" if h else '', str(hourly.get(h, 0)) if h else '']
        hr_data.append(row)
    story.append(_tbl(hr_data, [3*cm, 2*cm]*3, '#3d8b6b'))
    story.append(Spacer(1, 0.4*cm))

    # 前15名車牌
    if top_plates:
        story.append(Paragraph('■ 當日最常出現車牌（前15名）', head_sty))
        rows = [['車牌號碼', '次數', '最高信心', '方向']]
        for p in top_plates:
            dirs = (['進場'] if p[3] else []) + (['離場'] if p[4] else [])
            rows.append([p[0], str(p[1]), f"{p[2]*100:.0f}%",
                         '/'.join(dirs) if dirs else '—'])
        story.append(_tbl(rows, [5*cm, 3*cm, 3*cm, 4*cm], '#5b4f9e'))

    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        f'AI Camera Gate Monitor — 自動生成於 {now_str}', foot_sty))

    doc.build(story)
    size_kb = os.path.getsize(pdf_path) // 1024
    print(f"[REPORT] PDF generated: {pdf_path} ({size_kb} KB)")
    return pdf_path


def _smtp_connect(smtp_host, smtp_port, smtp_user, smtp_pass):
    """建立 SMTP 連線並登入，自動選擇 SSL(465) 或 STARTTLS(587/25)。"""
    import smtplib
    if smtp_port == 465:
        # SSL 直接連線
        server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15)
        server.ehlo()
    else:
        # STARTTLS（587 / 25）
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()
    server.login(smtp_user, smtp_pass)
    return server


def _translate_smtp_error(e: Exception) -> str:
    """將常見 SMTP 例外轉換為繁體中文說明。"""
    msg = str(e)
    if '534' in msg or '5.7.9' in msg:
        return ('Gmail 需要應用程式密碼（App Password）\n'
                '→ 請至 myaccount.google.com/apppasswords 產生 16 碼應用程式密碼，'
                '不可使用一般 Gmail 登入密碼。')
    if '535' in msg or '5.7.8' in msg:
        return '帳號或密碼錯誤（535）。請確認寄件帳號與應用程式密碼是否正確。'
    if '534' in msg and 'less secure' in msg:
        return '請至 Google 帳戶開啟「低安全性應用程式存取」或改用應用程式密碼。'
    if 'getaddrinfo' in msg or 'nodename' in msg.lower():
        return f'找不到 SMTP 伺服器 "{msg}"，請確認 SMTP 主機設定是否正確。'
    if 'timed out' in msg.lower() or 'timeout' in msg.lower():
        return f'連線逾時（Timeout）。請確認 SMTP 主機與 Port 是否正確，或防火牆是否允許對外發信。'
    if '421' in msg or '450' in msg:
        return f'SMTP 伺服器暫時拒絕連線（{msg[:60]}），請稍後再試。'
    if 'SSL' in msg or 'ssl' in msg:
        return f'SSL/TLS 連線失敗。請嘗試改用 Port 465（SSL）或 587（STARTTLS）。'
    return f'發送失敗：{msg}'


def send_report_email(pdf_path: str, date_str: str, recipient_override: str = '') -> tuple:
    """以 SMTP 發送每日報表 PDF 至設定的收件人。

    Args:
        pdf_path: PDF 檔案路徑
        date_str: 報表日期 'YYYY-MM-DD'
        recipient_override: 若提供，覆蓋資料庫中的收件人

    Returns:
        (success: bool, message: str)
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.base      import MIMEBase
    from email.mime.text      import MIMEText
    from email                import encoders

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings WHERE key LIKE 'email_%'")
    cfg = {r[0]: r[1] for r in cursor.fetchall()}
    conn.close()

    smtp_host = cfg.get('email_smtp_host', '')
    smtp_port = int(cfg.get('email_smtp_port', '587') or '587')
    smtp_user = cfg.get('email_smtp_user', '')
    smtp_pass = cfg.get('email_smtp_pass', '')
    recipient = recipient_override or cfg.get('email_recipient', '')

    # 檢查必填設定
    missing = []
    if not smtp_host: missing.append('SMTP 主機')
    if not smtp_user: missing.append('寄件帳號')
    if not smtp_pass: missing.append('應用程式密碼')
    if not recipient: missing.append('收件人 Email')
    if missing:
        return False, f'以下設定尚未填寫：{", ".join(missing)}。請至「設定 → Email 報表 → 進階 SMTP 設定」完成設定。'

    msg              = MIMEMultipart()
    msg['From']      = smtp_user
    msg['To']        = recipient
    msg['Subject']   = f'[AI車輛管制] 每日報表 {date_str}'
    msg.attach(MIMEText(
        f"您好，\n\n附件為 {date_str} 的車輛管制系統每日統計報表。\n\n此為系統自動發送，請勿回覆。\nAI Camera Gate Monitor",
        'plain', 'utf-8'))

    if os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        f'attachment; filename="report_{date_str}.pdf"')
        msg.attach(part)

    try:
        server = _smtp_connect(smtp_host, smtp_port, smtp_user, smtp_pass)
        server.sendmail(smtp_user, [recipient], msg.as_string())
        server.quit()
        print(f"[REPORT] Email sent to {recipient} for {date_str}")
        return True, f'✅ 報表已成功發送至 {recipient}'
    except Exception as e:
        print(f"[REPORT] Email error: {e}")
        return False, _translate_smtp_error(e)


def start_daily_report_scheduler():
    """啟動每日凌晨 00:05 自動產生報表並（若啟用）發送 Email 的背景執行緒。"""
    def _loop():
        import datetime as _dt
        while True:
            now    = _dt.datetime.now()
            target = now.replace(hour=0, minute=5, second=0, microsecond=0)
            if now >= target:
                target += _dt.timedelta(days=1)
            time.sleep((target - now).total_seconds())
            yesterday = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime('%Y-%m-%d')
            try:
                pdf_path = generate_daily_report_pdf(yesterday)
                conn   = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key='email_enabled'")
                res = cursor.fetchone(); conn.close()
                if res and res[0] == '1':
                    send_report_email(pdf_path, yesterday)
            except Exception as e:
                print(f"[REPORT] Daily scheduler error: {e}")
    threading.Thread(target=_loop, daemon=True, name="DailyReportScheduler").start()
    print("[SYSTEM] Daily report scheduler started (fires at 00:05 daily)")

# Background image save queue — offloads cv2.imwrite from the main detection loop
_image_save_queue = queue.Queue(maxsize=64)
def _image_saver_worker():
    while True:
        item = _image_save_queue.get()
        if item is None:
            break
        path, img = item
        try:
            # ── 截圖品質提升 ───────────────────────────────────────────────────
            # crop_ 檔案：原始裁切圖通常只有 ~80×30px。
            # 直接存 JPEG 放大後非常模糊。
            # 改進：放大到至少 300px 寬（最多 3×），然後 CLAHE + 銳化，
            # 再以 quality=90 存 JPEG，讓人工審查截圖清晰可辨識。
            # full_ 檔案：整張 1080p 幀，原始存即可。
            if os.path.basename(path).startswith("crop_"):
                h, w = img.shape[:2]
                if w > 0 and h > 0:
                    # 放大倍率：目標 300px 寬，上限 3×（避免過度放大失真）
                    scale = min(3.0, max(1.0, 300.0 / w))
                    if scale > 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    # CLAHE on luminance channel
                    try:
                        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                        y_ch, cr, cb = cv2.split(ycrcb)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                        y_ch = clahe.apply(y_ch)
                        img = cv2.cvtColor(cv2.merge([y_ch, cr, cb]), cv2.COLOR_YCrCb2BGR)
                    except Exception:
                        pass
                    # Unsharp mask
                    try:
                        blurred = cv2.GaussianBlur(img, (3, 3), 0)
                        img = cv2.addWeighted(img, 1.6, blurred, -0.6, 0)
                    except Exception:
                        pass
                cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            print(f"[SYSTEM] Image save error for {path}: {e}")
        finally:
            _image_save_queue.task_done()
_image_saver_thread = threading.Thread(target=_image_saver_worker, daemon=True)
_image_saver_thread.start()

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
gate_zone_masks = {}       # int cam_id -> pre-built 320x180 uint8 motion mask (改進 1)
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
    """Rebuild sv.PolygonZone objects and pre-build motion detection masks
    from the current GATE_ZONE_POLYGONS global.
    Called at startup and on every POST /api/zones — NOT on every frame."""
    global gate_zones, gate_zone_masks
    new_zones = {}
    new_masks = {}
    for cam_id, poly in GATE_ZONE_POLYGONS.items():
        new_zones[cam_id] = sv.PolygonZone(
            polygon=poly,
            triggering_anchors=[sv.Position.CENTER]
        )
        # 改進 1: Pre-build 320×180 motion mask once per zone change.
        # Main loop uses gate_zone_masks[cam_id] directly — no per-frame
        # np.zeros() allocation or fillPoly() call needed.
        mask = np.zeros((180, 320), dtype=np.uint8)
        poly_small = (poly / 6.0).astype(np.int32)
        cv2.fillPoly(mask, [poly_small], 255)
        new_masks[cam_id] = mask
    gate_zones = new_zones
    gate_zone_masks = new_masks
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

    # DB migration: direction column (ENTER/EXIT/UNKNOWN)
    try:
        cursor.execute("ALTER TABLE detections ADD COLUMN direction TEXT DEFAULT 'UNKNOWN'")
    except sqlite3.OperationalError:
        pass
        
    # 改進 6: Only classify detections where vehicle_type is missing/unknown.
    # Previous: SELECT * loads every row on every startup (slow at scale).
    # Now: only rows that genuinely need updating are fetched.
    # After first run, this query returns 0 rows and completes instantly.
    cursor.execute("""
        SELECT id, plate_number FROM detections
        WHERE vehicle_type IS NULL OR vehicle_type = '' OR vehicle_type = 'UNKNOWN'
    """)
    rows_to_fix = cursor.fetchall()
    if rows_to_fix:
        updates = [(classify_vehicle_type(plate), row_id) for row_id, plate in rows_to_fix]
        cursor.executemany("UPDATE detections SET vehicle_type = ? WHERE id = ?", updates)
        conn.commit()
        print(f"Retroactively classified {len(updates)} detections.")

 
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
    # Email report settings
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_enabled', '0')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_smtp_host', 'smtp.gmail.com')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_smtp_port', '587')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_smtp_user', '')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_smtp_pass', '')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('email_recipient', '')")

    # ── 改進 2: Performance-critical indices (none existed before) ─────────────
    # Without these, every API query does a full O(n) table scan.
    # With these, common queries become O(log n) — 10-50x faster at 10k+ rows.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_timestamp  ON detections(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_plate       ON detections(plate_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_det_camera      ON detections(camera_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_plate     ON alerts(plate_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alerts(timestamp DESC)")

    conn.commit()
    conn.close()

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
        # 改進 10: Poll go2rtc API for readiness instead of fixed sleep(2.0).
        # go2rtc is typically ready in 0.3–0.8s; max wait 3s (30 × 100ms).
        _ready = False
        for _ in range(30):
            time.sleep(0.1)
            try:
                urllib.request.urlopen("http://127.0.0.1:1984/api", timeout=0.3)
                _ready = True
                break
            except Exception:
                continue
        print("[SYSTEM] go2rtc started successfully." if _ready else "[SYSTEM] go2rtc started (readiness timeout, proceeding anyway).")
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
    # 改進 11: Reuse db_conn_persistent (check_same_thread=False) instead of
    # opening a new sqlite3 connection on every sync call (~every 30s).
    try:
        cursor = db_conn_persistent.cursor()
        cursor.execute("SELECT id, name, rtsp_url FROM cameras WHERE is_active = 1")
        active_cameras = cursor.fetchall()
    except Exception:
        # Fallback: open a fresh connection if persistent one is unavailable
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
    with readers_lock:
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
            with readers_lock:
                active_readers[cam_id] = reader

# YOLO Model Path
MODEL_PATH = "license-plate-finetune-v1n_openvino_model"  # YOLOv11n OpenVINO — 2x faster than v8n (58ms vs 118ms/frame on CPU)

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
        self._skip_counter = 0           # init here, not inside hot read loop
        self._offline_threshold_min = 5  # cached DB setting; refreshed on each offline tick
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
            # Refresh cached threshold from DB on each offline tick (not every frame).
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = 'offline_threshold_minutes'")
                row = cursor.fetchone()
                self._offline_threshold_min = int(row[0]) if row else 5
                conn.close()
            except Exception:
                pass  # keep previous cached value

            if duration >= self._offline_threshold_min * 60 and not self.alert_sent:
                self.alert_sent = True
                alert_text = f"⚠️ [相機離線警報] ⚠️\n🎥 攝影機：{self.name} (ID: {self.cam_id})\n❌ 狀態：已失去連線超過 {self._offline_threshold_min} 分鐘！\n請儘速檢查網路線路或攝影機電源。"
                print(f"[SYSTEM] {alert_text}")
                send_telegram_text_async(alert_text)

    def _reader(self):
        while self.running:
            print(f"Connecting to RTSP proxy for camera {self.name}...")
            current_url = self.url
            open_url = self.proxy_url

            # OPENCV_FFMPEG_CAPTURE_OPTIONS already set at module startup via os.environ.
            # Removed: import os + os.environ reassignment on every reconnect.
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
                # skip FIRST, sleep only for kept frames (saves 80ms per discarded frame).
                self._skip_counter += 1
                if self._skip_counter % 2 == 0:
                    continue  # discard frame without sleeping

                time.sleep(0.08)  # pace kept frames to ~6fps

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

# ── Module-level constants for clean_plate_text (改進 7) ─────────────────────
# Moved out of function body so they are created once, not on every OCR call.
_PLATE_RE        = re.compile(r'[^A-Za-z0-9]')
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
_LETTER_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'J': '1',
                    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'Y': '7', 'A': '4'}
_FORMAT_SPLITS   = {5: 2, 6: 3, 7: 3, 8: 4}

# Clean up recognized plate text to standard format (Alphanumeric uppercase, no hyphens)
def clean_plate_text(text):
    cleaned = _PLATE_RE.sub('', text).upper()

    # ── Improvement 2: Enhanced position-based character correction ──────────
    # Taiwan license plate formats and their split points:
    #   5-char: 2L + 3D  (e.g. AB-123)   → split=2
    #   6-char: 3L + 3D  (e.g. ABC-123)  → split=3
    #   7-char: 3L + 4D  (e.g. ABC-1234) → split=3
    #   8-char: 4L + 4D  (e.g. ABCD-1234)→ split=4  (electric scooter)
    # In the letter-zone, digits that look like letters are corrected to letters.
    # In the digit-zone, letters that look like digits are corrected to digits.
    if len(cleaned) in _FORMAT_SPLITS:
        split = _FORMAT_SPLITS[len(cleaned)]
        prefix = cleaned[:split]
        suffix = cleaned[split:]

        # Only apply if the prefix 'looks like' a letter zone and suffix 'looks like' a digit zone
        letter_zone_score = sum(c.isalpha() or c in _DIGIT_TO_LETTER for c in prefix)
        digit_zone_score  = sum(c.isdigit() or c in _LETTER_TO_DIGIT for c in suffix)

        if letter_zone_score >= split - 1 and digit_zone_score >= len(suffix) - 1:
            corrected = [_DIGIT_TO_LETTER.get(c, c) for c in prefix] + \
                        [_LETTER_TO_DIGIT.get(c, c) for c in suffix]
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
        # datetime imported at module top — no inline import needed.
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
        # 改進 12: Use module-level credential cache — no DB open on each call.
        # Cache is loaded once at startup and invalidated when credentials change.
        if not _auth_cache["loaded"]:
            reload_auth_cache()
        with _auth_cache_lock:
            username = _auth_cache["username"]
            password = _auth_cache["password"]

        if not auth_header:
            return False
        if not auth_header.startswith('Basic '):
            return False
        # base64 imported at module level (改進 21)
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
                       d.direction,
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

        # ── Report API ──────────────────────────────────────────────────────────
        elif path == "/api/report/download":
            # Generate (or re-use cached) PDF and stream it to the browser
            date_str = query.get('date', [datetime.datetime.now().strftime('%Y-%m-%d')])[0].strip()
            # Basic date format validation
            import re as _re
            if not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                self.send_response(400)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(b'Invalid date format (expect YYYY-MM-DD)')
                return
            try:
                pdf_path = generate_daily_report_pdf(date_str)
                self.send_response(200)
                self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Disposition',
                                 f'attachment; filename="report_{date_str}.pdf"')
                self.send_header('Content-Length', str(os.path.getsize(pdf_path)))
                self.send_cors_headers()
                self.end_headers()
                with open(pdf_path, 'rb') as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

        elif path == "/api/report/send":
            # Trigger immediate email send for a given date
            date_str = query.get('date', [datetime.datetime.now().strftime('%Y-%m-%d')])[0].strip()
            try:
                pdf_path = generate_daily_report_pdf(date_str)
                ok, msg  = send_report_email(pdf_path, date_str)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'success': ok, 'message': msg}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode())

        elif path == "/api/settings/email":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            conn   = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings WHERE key LIKE 'email_%'")
            cfg = {r[0]: r[1] for r in cursor.fetchall()}
            conn.close()
            self.wfile.write(json.dumps({
                'email_enabled':   cfg.get('email_enabled',   '0'),
                'email_smtp_host': cfg.get('email_smtp_host', 'smtp.gmail.com'),
                'email_smtp_port': cfg.get('email_smtp_port', '587'),
                'email_smtp_user': cfg.get('email_smtp_user', ''),
                'email_smtp_pass': cfg.get('email_smtp_pass', ''),
                'email_recipient': cfg.get('email_recipient', ''),
            }).encode())

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
                reload_auth_cache()  # 改進 12: refresh cache so new creds take effect immediately
                
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

        elif path == "/api/settings/email/save":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                conn   = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                for key in ('email_enabled', 'email_smtp_host', 'email_smtp_port',
                            'email_smtp_user', 'email_smtp_pass', 'email_recipient'):
                    if key in data:
                        cursor.execute(
                            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                            (key, str(data[key])))
                conn.commit(); conn.close()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

        elif path == "/api/report/test-smtp":
            # 測試 SMTP 連線（不實際發信，只驗證帳號密碼）
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data      = json.loads(post_data)
                smtp_host = data.get('email_smtp_host', '').strip()
                smtp_port = int(data.get('email_smtp_port', 587) or 587)
                smtp_user = data.get('email_smtp_user', '').strip()
                smtp_pass = data.get('email_smtp_pass', '').strip()
                if not all([smtp_host, smtp_user, smtp_pass]):
                    raise ValueError('SMTP 主機、帳號、密碼均為必填')
                server = _smtp_connect(smtp_host, smtp_port, smtp_user, smtp_pass)
                server.quit()
                ok  = True
                msg = f'✅ SMTP 連線成功！帳號 {smtp_user} 驗證通過，可以正常發信。'
            except Exception as e:
                ok  = False
                msg = _translate_smtp_error(e)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'success': ok, 'message': msg}).encode())

    def do_PUT(self):
        """Handle PUT /api/watchlist — edit an existing watchlist entry."""
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

        if path == "/api/watchlist":
            content_length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(raw)
                original_plate = clean_plate_text(data.get('original_plate', '').strip())
                new_plate      = clean_plate_text(data.get('plate_number', '').strip())
                category       = data.get('category', '').strip().upper()
                description    = data.get('description', '').strip()

                if not original_plate or not new_plate or not category:
                    raise ValueError("original_plate, plate_number, category 為必填欄位")

                conn   = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                if original_plate != new_plate:
                    # 車牌號碼已修改：先刪除舊記錄，再插入新記錄
                    cursor.execute("DELETE FROM watchlist WHERE plate_number = ?", (original_plate,))
                    cursor.execute("""
                        INSERT INTO watchlist (plate_number, category, description)
                        VALUES (?, ?, ?)
                    """, (new_plate, category, description))
                else:
                    # 只更新 category / description，保留 created_at
                    cursor.execute("""
                        UPDATE watchlist SET category=?, description=?
                        WHERE plate_number=?
                    """, (category, description, original_plate))

                conn.commit()
                conn.close()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "plate_number": new_plate}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

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

    # Start Daily Report Scheduler (fires at 00:05 daily)
    start_daily_report_scheduler()

    # Load YOLO Model with OpenVINO on CPU
    # (Intel GPU compiler has bugs with certain layers, causing CISA routine errors and crashes)
    print(f"Loading YOLOv11n model from {MODEL_PATH} (device=cpu)...")
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
    
    # Load PaddleOCR (v3.x API)
    # - use_doc_orientation_classify=False : skip PP-LCNet doc-orientation model (not needed for plates)
    # - use_doc_unwarping=False            : skip UVDoc document-unwarping model (not needed for plates)
    # - use_textline_orientation=False     : skip textline-orientation model (plates are horizontal)
    # - text_recognition_batch_size=1      : replaces deprecated rec_batch_num
    # Result: only 2 models loaded (PP-OCRv6_medium_det + PP-OCRv6_medium_rec)
    # Note: det=False (skip detection stage) is not available in PaddleOCR v3.x API.
    print("Initializing PaddleOCR v3.x (det+rec only, no doc preprocessor)...")
    ocr = PaddleOCR(
        lang='en',
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=1,
    )

    # ── 改進 8: Cache CLAHE object (created once, reused every frame) ─────────
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Start all active RTSP readers from database
    sync_active_readers()

    # Start HTTP Web Server Thread
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()

    # ── Improvement 4: sv.ByteTrack per-camera vehicle tracker ──────────────
    # Replaces the manual IoU-based tracked_vehicles list.
    # ByteTrack assigns a persistent tracker_id to each plate detection across
    # frames using Kalman filtering + IoU, giving more robust identity across
    # the 30-frame trigger window even when the vehicle is moving.
    byte_trackers = {}   # cam_id -> sv.ByteTrack instance
    tracked_by_id = {}   # cam_id -> {tracker_id(int): vehicle_record_dict}
    TRACK_LOST_SECONDS = 300.0   # Keep record for 5 min (same as old STATIONARY_TIMEOUT)
    # Legacy tracked_vehicles kept as empty dict so any stray references don't crash
    tracked_vehicles = {}

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

    # Persistent SQLite connection with WAL mode for lower per-commit overhead.
    # All DB writes in the main loop use this connection + db_write_lock.
    # 改進 11: Declared global so sync_active_readers() (module-level) can reuse it.
    global db_conn_persistent, db_write_lock
    db_conn_persistent = sqlite3.connect(DB_PATH, check_same_thread=False)
    db_conn_persistent.execute("PRAGMA journal_mode=WAL")
    db_conn_persistent.execute("PRAGMA synchronous=NORMAL")
    db_conn_persistent.execute("PRAGMA cache_size=-8000")  # 8MB page cache
    # db_write_lock already declared at module level; global statement above ensures
    # any re-assignment here updates the same object referenced by other functions.

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
    track_y_history  = {}   # cam_id -> {tracker_id: [center_y, ...]} — for direction detection

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

            _loop_start = time.time()  # 改進 7: track loop duration for sleep budgeting
            processed_any = False
            with readers_lock:
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
                    
                    # ── Improvement: Polygon-masked driveway motion detection ──────
                    # Instead of a hardcoded box crop, we create a 320x180 black mask,
                    # scale down the custom gate polygon (GATE_ZONE_POLYGONS) by 6x,
                    # draw it on the mask, and bitwise-AND it with diff_thresh.
                    # This ensures motion is ONLY evaluated inside the custom polygon zone,
                    # completely ignoring background wind/leaves noise outside the zone.
                    # 改進 1: Use pre-cached mask from gate_zone_masks (built once in
                    # rebuild_gate_zones). No np.zeros() or fillPoly() per frame.
                    raw_motion_detected = False
                    cached_mask = gate_zone_masks.get(cam_id)
                    if cached_mask is not None:
                        # Apply pre-built mask directly
                        driveway_diff = cv2.bitwise_and(diff_thresh, cached_mask)
                        changed_driveway_pixels = cv2.countNonZero(driveway_diff)

                        # Motion threshold by cam_id (改進 9 from report: avoid cam_name string)
                        # Camera 002 (cam_id=1) threshold raised 100→200: Camera 002 was triggering
                        # every ~8s from wind/shadow environmental noise within the zone polygon.
                        # Vehicles produce 300-800+ changed pixels; leaf/shadow noise is 80-150px.
                        _MOTION_THRESHOLDS = {1: 200, 3: 120, 6: 100}
                        motion_threshold = _MOTION_THRESHOLDS.get(cam_id, 100)

                        if changed_driveway_pixels >= motion_threshold:
                            raw_motion_detected = True
                    else:
                        # Fallback to full frame motion detection if no polygon is set
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

                # ── 光照量測：每幀更新，Zone 區域 mean/std ────────────────────
                # 使用已計算的 gray_small (320×180) 與 cached_mask，幾乎零額外成本。
                _bright_src = gray_small
                _bright_mask = gate_zone_masks.get(cam_id)
                if _bright_mask is not None:
                    _zone_px = _bright_src[_bright_mask > 0]
                else:
                    _zone_px = _bright_src.flatten()
                if len(_zone_px) > 50:
                    _bm = float(np.mean(_zone_px))
                    _bs = float(np.std(_zone_px))
                    if cam_id not in _cam_brightness_history:
                        _cam_brightness_history[cam_id] = deque(maxlen=60)
                    _cam_brightness_history[cam_id].append((_bm, _bs))
                    _hist = _cam_brightness_history[cam_id]
                    if len(_hist) >= 10:  # 需至少 10 個樣本才開始調整
                        _avg_m = sum(x[0] for x in _hist) / len(_hist)
                        _avg_s = sum(x[1] for x in _hist) / len(_hist)
                        _new_conf, _cond = _compute_yolo_conf(_avg_m, _avg_s)
                        _old_conf = _cam_yolo_conf.get(cam_id, 0.15)
                        _cam_yolo_conf[cam_id] = _new_conf
                        _cam_lighting_condition[cam_id] = _cond
                        # 每 60 秒記錄一次，或光照狀態改變時立即記錄
                        _now_l = time.time()
                        _cond_changed = abs(_new_conf - _old_conf) > 0.019
                        if _cond_changed or (_now_l - _cam_last_lighting_log.get(cam_id, 0)) > 60:
                            print(f"[LIGHTING] {active_camera_names.get(cam_id, cam_id)} "
                                  f"mean={_avg_m:.0f} std={_avg_s:.0f} "
                                  f"conf={_new_conf:.2f} [{_cond}]")
                            _cam_last_lighting_log[cam_id] = _now_l

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

                # 改進 4: Lazy display_frame creation — avoid unconditional 6MB frame.copy().
                # For early-exit paths (paused / no-motion), only copy when do_display_update.
                # For the YOLO path (has_motion + active), display_frame is always needed
                # for bbox annotation, so it is created below after these checks.
                do_display_update = (time.time() - last_display_update.get(cam_id, 0)) >= DISPLAY_UPDATE_INTERVAL
                # DISPLAY_W / DISPLAY_H are module-level constants (改進 14)


                # 改進 14: _make_display_frame and _encode_and_push are now module-level
                # functions (defined once at import time). Previous inline `def` created
                # new function objects + closures on every frame (18/s for 3 cameras).

                # Check if LPR is paused for this specific camera
                camera_lpr_active = True
                if manual_override == 'paused':
                    camera_lpr_active = False
                elif cam_name == "學校大門":
                    camera_lpr_active = system_enabled

                if not camera_lpr_active:
                    if do_display_update:
                        display_frame = _make_display_frame(frame, cam_name, cam_id)
                        cv2.putText(display_frame, "SYSTEM PAUSED (MONITORING INACTIVE)", (30, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        _encode_and_push(display_frame, cam_id, last_display_update)
                    continue

                if not has_motion:
                    if do_display_update:
                        _encode_and_push(_make_display_frame(frame, cam_name, cam_id), cam_id, last_display_update)
                    continue

                # Has motion + camera active: always need display_frame for YOLO bbox annotation.
                display_frame = _make_display_frame(frame, cam_name, cam_id)

                
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
                        # First frame of window: reset candidate buffer + Y history for this camera
                        plate_candidates[cam_id] = []
                        track_y_history[cam_id]  = {}
                        print(f"[SYSTEM] {cam_name} driveway motion trigger: running LPR detection (30 frames).")
                    if cam_trigger_frames == 0:
                        trigger_window_closing = True  # Window just expired → commit best plate

                # Pre-YOLO display update: push current frame to stream NOW, before
                # the blocking model() call occupies CPU for ~200ms. This keeps the
                # MJPEG stream showing a fresh frame instead of a stale frozen one.
                # 改進 14: Use shared _encode_and_push() — display_w/display_h are DISPLAY_W/DISPLAY_H.
                if do_display_update:
                    _encode_and_push(display_frame, cam_id, last_display_update)

                if run_lpr:
                    # Mask out the timestamp watermark in the bottom-right corner to prevent false detections
                    detection_frame = frame.copy()
                    h, w, _ = detection_frame.shape
                    detection_frame[int(h*0.9):h, int(w*0.7):w] = 0
                    
                    # ── 改進 6: YOLO ROI derived from PolygonZone bounding box ──
                    # Dynamically crop to the polygon's bounding box (+ 5% margin)
                    # instead of hardcoded camera-name branches.
                    # Smaller YOLO input → faster inference (27ms → ~15ms when zone
                    # covers ~50% of frame area).
                    crop_y1, crop_y2, crop_x1, crop_x2 = 0, h, 0, w
                    _zone_poly = GATE_ZONE_POLYGONS.get(cam_id)
                    if _zone_poly is not None and len(_zone_poly) > 0:
                        _margin = 0.05
                        _x_min = int(_zone_poly[:, 0].min())
                        _y_min = int(_zone_poly[:, 1].min())
                        _x_max = int(_zone_poly[:, 0].max())
                        _y_max = int(_zone_poly[:, 1].max())
                        crop_x1 = max(0, int(_x_min * (1 - _margin)))
                        crop_y1 = max(0, int(_y_min * (1 - _margin)))
                        crop_x2 = min(w, int(_x_max * (1 + _margin)))
                        crop_y2 = min(h, int(_y_max * (1 + _margin)))

                    yolo_input = detection_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    # 光照自適應 conf：由 _cam_yolo_conf 動態提供（預設 0.15）。
                    # 每幀在 Zone 區域量測亮度並滾動平均後計算：
                    #   overexposed(炫光) → 0.12, bright(晴天) → 0.20,
                    #   normal(一般) → 0.15, dim(黃昏) → 0.12, dark(夜間) → 0.10
                    # OCR 信心度閾值 (0.40/0.50/0.65) 仍負責最終品質過濾。
                    # iou=0.45: 積極抑制重複框。改進 5: Dynamic imgsz.
                    _crop_long = max(crop_x2 - crop_x1, crop_y2 - crop_y1)
                    _imgsz = max(320, min(640, (_crop_long // 32) * 32))
                    _yolo_conf = _cam_yolo_conf.get(cam_id, 0.15)
                    results = model(yolo_input, conf=_yolo_conf, iou=0.45, imgsz=_imgsz, verbose=False)
                    current_time = time.time()
                    
                    # Check if any plates were found (result boxes are relative to cropped image)
                    any_plates_detected = False
                    for result in results:
                        if len(result.boxes) > 0:
                            any_plates_detected = True
                            break
                    if any_plates_detected:
                        last_plate_times[cam_id] = current_time
                    
                    if cam_id not in recently_logged_plates:
                        recently_logged_plates[cam_id] = {}

                    # tracked_vehicles always empty after ByteTrack replaced IoU tracker — cleanup removed.
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

                    # ── Improvement 4: sv.ByteTrack – assign persistent tracker_id ──
                    # Each plate detection gets a stable ID across the 30-frame trigger
                    # window, enabling per-vehicle candidate grouping at window close.
                    if cam_id not in byte_trackers:
                        byte_trackers[cam_id] = sv.ByteTrack(
                            track_activation_threshold=0.15,  # aligned with YOLO conf=0.15
                            lost_track_buffer=90,             # ~30s at 3fps YOLO cadence
                            minimum_matching_threshold=0.5,
                            frame_rate=3,
                        )
                        print(f"[SYSTEM] ByteTrack initialized for camera {cam_id} ({cam_name})")
                    if len(filtered_detections) > 0:
                        tracked_dets = byte_trackers[cam_id].update_with_detections(filtered_detections)
                    else:
                        tracked_dets = sv.Detections.empty()

                    # ── Iterate over ByteTrack-labelled plate detections ──────────
                    for det_idx in range(len(tracked_dets)):
                        xyxy = tracked_dets.xyxy[det_idx]
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        # Keep the legacy top-left corner filter as backup
                        # (camera name overlay area – already outside zone polygon for most cameras)
                        if x1 < 400 and y1 < 80:
                            continue

                        # Extract ByteTrack tracker_id (-1 if not available)
                        tracker_id = int(tracked_dets.tracker_id[det_idx]) \
                            if tracked_dets.tracker_id is not None else -1

                        confidence = float(tracked_dets.confidence[det_idx])

                        # ── 方向判斷：累積每幀車牌中心點 Y 座標 ─────────────────────
                        # 每幀 ByteTrack 成功追蹤時（無論 OCR 是否通過）均記錄 center_y
                        _center_y = (y1 + y2) // 2
                        if cam_id not in track_y_history:
                            track_y_history[cam_id] = {}
                        if tracker_id not in track_y_history[cam_id]:
                            track_y_history[cam_id][tracker_id] = []
                        track_y_history[cam_id][tracker_id].append(_center_y)

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

                        # ── 強化 PRE-1: Laplacian 模糊評分 ────────────────────────────────
                        # 計算灰階 Laplacian 方差，低於閾值代表過度模糊，拒絕送 OCR
                        # 節省 OCR 資源，避免模糊幀污染 ocr_history
                        _gray_blur_check = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                        _lap_var = cv2.Laplacian(_gray_blur_check, cv2.CV_64F).var()
                        BLUR_THRESHOLD = 30.0  # 低於此值 = 過度模糊，跳過
                        if _lap_var < BLUR_THRESHOLD:
                            continue  # 模糊幀，不送 OCR

                        # 加入邊框（replicate 模式避免邊緣 artifact）
                        cropped_plate_padded = cv2.copyMakeBorder(
                            cropped_plate,
                            top=10, bottom=10, left=15, right=15,
                            borderType=cv2.BORDER_REPLICATE
                        )

                        # ── 強化 PRE-2: 放大至 128px 高（原本 64px） ─────────────────────
                        # 更高解析度有助於小字符（機車牌、遠距離）的 OCR 準確率
                        target_h = 128  # 改進：128px（原本 64px）
                        ph, pw, _ = cropped_plate_padded.shape
                        target_w = int(pw * (target_h / ph))
                        enhanced = cv2.resize(cropped_plate_padded, (target_w, target_h),
                                              interpolation=cv2.INTER_LANCZOS4)  # Lanczos > Cubic 於放大

                        # ── 強化 PRE-3: CLAHE 對比度強化（亮度通道）────────────────────
                        # (改進 8: reuse cached _clahe object — no allocation per detection)
                        try:
                            ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
                            y_channel, cr, cb = cv2.split(ycrcb)
                            y_channel = _clahe.apply(y_channel)
                            ycrcb = cv2.merge([y_channel, cr, cb])
                            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                        except Exception as e:
                            print(f"[SYSTEM] CLAHE enhancement failed: {e}")

                        # ── 強化 PRE-4: Unsharp Mask 銳化 ────────────────────────────────
                        try:
                            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                            enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
                        except Exception as e:
                            print(f"[SYSTEM] Sharpening failed: {e}")

                        # ── 強化 PRE-5: 低對比度時啟用自適應二值化備援 ───────────────────
                        # 強光/逆光場景下，車牌字符與背景對比度低（< 40）。
                        # 此時以 CLAHE + Otsu 二值化後轉灰階 BGR，與彩色版各送 OCR，
                        # 採用 ocr_conf 較高的結果。
                        _gray_for_contrast = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                        _contrast = float(_gray_for_contrast.std())
                        enhanced_binarized = None
                        if _contrast < 40.0:
                            try:
                                # CLAHE 強化後 Otsu 二值化
                                _clahe_tmp = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                                _y_bin = _clahe_tmp.apply(_gray_for_contrast)
                                _, _bin = cv2.threshold(_y_bin, 0, 255,
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                # 轉回 BGR 3 通道（PaddleOCR 期望彩色輸入）
                                enhanced_binarized = cv2.cvtColor(_bin, cv2.COLOR_GRAY2BGR)
                            except Exception as e:
                                print(f"[SYSTEM] Binarize fallback failed: {e}")

                        # Call PaddleOCR predict (replaces deprecated ocr.ocr())
                        ocr_res = ocr.predict(enhanced)

                        plate_text = ""
                        ocr_conf = 0.0
                        
                        if ocr_res and len(ocr_res) > 0:
                            res_dict = ocr_res[0]  # OCRResult dict-like object
                            texts = res_dict.get('rec_texts', [])
                            scores = res_dict.get('rec_scores', [])
                            plate_text = "".join(texts)
                            # 改進 10: Use MEAN score (not max) to reflect full-plate
                            # recognition quality. max() was too easily gamed by a single
                            # high-confidence character in an otherwise blurry read.
                            ocr_conf = (sum(scores) / len(scores)) if scores else 0.0

                        # ── 強化 PRE-5 比對：若二值化備援啟用，與彩色版比較取優 ──────────
                        if enhanced_binarized is not None:
                            try:
                                ocr_res_bin = ocr.predict(enhanced_binarized)
                                if ocr_res_bin and len(ocr_res_bin) > 0:
                                    res_bin = ocr_res_bin[0]
                                    texts_bin  = res_bin.get('rec_texts', [])
                                    scores_bin = res_bin.get('rec_scores', [])
                                    plate_text_bin = "".join(texts_bin)
                                    ocr_conf_bin   = (sum(scores_bin) / len(scores_bin)) if scores_bin else 0.0
                                    if ocr_conf_bin > ocr_conf:
                                        # 二值化版本更佳，採用之
                                        plate_text = plate_text_bin
                                        ocr_conf   = ocr_conf_bin
                                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                              f"[BINARIZE] contrast={_contrast:.1f} → bin OCR better "
                                              f"({ocr_conf_bin:.2f} > {ocr_conf:.2f}) '{plate_text_bin}'")
                            except Exception as _e:
                                pass  # 二值化版失敗，沿用彩色版

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

                            # Only keep the best candidate per tracker_id to limit RAM usage.
                            # Full 1080p frames are expensive; we only store the top-scoring one.
                            if cam_id not in plate_candidates:
                                plate_candidates[cam_id] = []

                            existing_idx = next(
                                (i for i, c in enumerate(plate_candidates[cam_id])
                                 if c["tracker_id"] == tracker_id),
                                None
                            )
                            if existing_idx is None:
                                # 改進 OCR-VOTE: 初始化 ocr_history，儲存輕量字串歷史
                                # （不儲存 frame/crop，純字串+conf，幾乎零 RAM 開銷）
                                plate_candidates[cam_id].append({
                                    "plate":         cleaned_plate,
                                    "tracker_id":    tracker_id,
                                    "yolo_conf":     confidence,
                                    "ocr_conf":      ocr_conf,
                                    "combined":      combined_conf,
                                    "cropped_plate": cropped_plate.copy(),
                                    # 改進 13: Store frame reference (not .copy()) during accumulation.
                                    # Only the WINNER gets frame.copy() at trigger_window_closing time.
                                    "full_frame":    frame,
                                    "box":           [x1, y1, x2, y2],
                                    "timestamp_str": timestamp_str,
                                    "ocr_history":   [(cleaned_plate, ocr_conf)],
                                })
                            else:
                                existing = plate_candidates[cam_id][existing_idx]
                                # 改進 OCR-VOTE: 無論 conf 高低，都將本幀 OCR 加入 history
                                existing["ocr_history"].append((cleaned_plate, ocr_conf))
                                if combined_conf > existing["combined"]:
                                    # Better candidate found: update frame/crop in place
                                    existing["plate"]         = cleaned_plate
                                    existing["yolo_conf"]     = confidence
                                    existing["ocr_conf"]      = ocr_conf
                                    existing["combined"]      = combined_conf
                                    existing["cropped_plate"] = cropped_plate.copy()
                                    existing["full_frame"]    = frame  # 改進 13: ref only
                                    existing["box"]           = [x1, y1, x2, y2]
                                    existing["timestamp_str"] = timestamp_str
                                # else: frame/crop already best; only ocr_history updated above
                            unique_tids = len(plate_candidates[cam_id])
                            _hist_len = len(plate_candidates[cam_id][
                                next(i for i, c in enumerate(plate_candidates[cam_id])
                                     if c["tracker_id"] == tracker_id)
                            ]["ocr_history"])
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CANDIDATE] {cam_name}: {cleaned_plate} "
                                  f"(YOLO={confidence:.2f} OCR={ocr_conf:.2f} combined={combined_conf:.2f}) "
                                  f"tid={tracker_id} hist={_hist_len} [{unique_tids} unique tracker(s) buffered]")

                # ── Commit best plate(s) when trigger window closes ───────────
                if trigger_window_closing:
                    candidates = plate_candidates.get(cam_id, [])
                    if candidates:
                        # ── Group candidates by ByteTrack tracker_id ─────────
                        # Each unique tracker_id = one physical vehicle.
                        # This allows multiple vehicles passing simultaneously
                        # to each get their own DB record.
                        by_track = defaultdict(list)  # defaultdict imported at module top
                        for c in candidates:
                            by_track[c.get("tracker_id", -1)].append(c)

                        plate_candidates[cam_id] = []  # Reset buffer
                        track_y_history[cam_id]  = {}  # Reset direction history

                        # Init per-camera tracking dict
                        if cam_id not in tracked_by_id:
                            tracked_by_id[cam_id] = {}
                        if cam_id not in recently_logged_plates:
                            recently_logged_plates[cam_id] = {}

                        current_time = time.time()
                        # Expire stale track records
                        tracked_by_id[cam_id] = {
                            tid: rec for tid, rec in tracked_by_id[cam_id].items()
                            if current_time - rec["last_seen"] < TRACK_LOST_SECONDS
                        }
                        recently_logged_plates[cam_id] = {
                            k: v for k, v in recently_logged_plates[cam_id].items()
                            if current_time - v < LOGGED_SUPPRESSION_TIMEOUT
                        }

                        # ── Per tracker_id: commit best candidate ─────────────
                        # 改進 OCR-VOTE: 字符級加權投票取代舊有的 +0.15 頻次加成。
                        # 由於每個 tracker_id 已在 ocr_history 中累積所有幀讀取，
                        # vote_plate_text() 可對每個字符位置獨立投票，有效消除
                        # 偶發性誤讀（0→O、1→I、B→8 等常見 OCR 混淆字符）。
                        for tid, track_cands in by_track.items():
                            best = max(track_cands, key=lambda c: c["combined"])

                            # 收集此 tracker_id 的完整 OCR 歷史（跨所有幀）
                            all_ocr_history = []
                            for c in track_cands:
                                all_ocr_history.extend(c.get("ocr_history", [(c["plate"], c["ocr_conf"])]))

                            # 字符級投票產生最終車牌
                            voted_plate = vote_plate_text(all_ocr_history)
                            raw_best    = best["plate"]  # 保留原始最佳供 log 比對

                            # 若投票結果非空且合理長度，採用投票結果
                            if voted_plate and 3 <= len(voted_plate) <= 10:
                                cleaned_plate = voted_plate
                            else:
                                cleaned_plate = raw_best  # fallback

                            # ── 方向判斷：由 Y 座標歷史計算 ENTER/EXIT ────────────
                            y_hist    = track_y_history.get(cam_id, {}).get(tid, [])
                            direction = compute_direction(y_hist)

                            ocr_conf      = best["ocr_conf"]
                            confidence_b  = best["yolo_conf"]
                            cropped_plate = best["cropped_plate"]
                            frame_b       = best["full_frame"].copy()  # 改進 13: copy ONLY the winner
                            box_b         = best["box"]
                            timestamp_str = best["timestamp_str"]
                            x1, y1, x2, y2 = box_b

                            vote_note = f"voted='{voted_plate}'" if voted_plate != raw_best else "voted=same"
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BEST-PLATE] {cam_name}: "
                                  f"tid={tid} '{cleaned_plate}' hist={len(all_ocr_history)} {vote_note} dir={direction} "
                                  f"(YOLO={confidence_b:.2f} OCR={ocr_conf:.2f})")

                            # ── Duplicate check 1: ByteTrack tracker_id lookup ─
                            is_duplicate  = False
                            matched_rec   = tracked_by_id[cam_id].get(tid)

                            if matched_rec:
                                # Same physical vehicle seen again – update record
                                matched_rec["last_seen"] = current_time
                                is_duplicate = True
                                # If better OCR → update DB record
                                if cleaned_plate != matched_rec["plate"] and ocr_conf > matched_rec.get("confidence", 0.0):
                                    old_plate = matched_rec["plate"]
                                    det_id    = matched_rec.get("detection_id")
                                    new_v_type = classify_vehicle_type(cleaned_plate)
                                    if det_id:
                                        try:
                                            with db_write_lock:
                                                cursor  = db_conn_persistent.cursor()
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
                                                db_conn_persistent.commit()
                                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TRACK] Updated tid={tid} on {cam_name}: {old_plate} -> {cleaned_plate} (Conf: {round(ocr_conf, 2)})")
                                        except Exception as e:
                                            print(f"Error updating tracked plate: {e}")
                                    matched_rec["plate"]        = cleaned_plate
                                    matched_rec["confidence"]   = ocr_conf
                                    matched_rec["vehicle_type"] = classify_vehicle_type(cleaned_plate)
                                    recently_logged_plates[cam_id][cleaned_plate] = current_time

                            # ── Duplicate check 2: string similarity fallback ──
                            # 改進 8: Also verify the matched entry is still within
                            # LOGGED_SUPPRESSION_TIMEOUT — cleanup only runs per trigger
                            # window, so stale entries may linger briefly between cleanups.
                            if not is_duplicate:
                                for logged_plate, logged_time in recently_logged_plates[cam_id].items():
                                    if (current_time - logged_time) < LOGGED_SUPPRESSION_TIMEOUT \
                                            and is_similar_plate(cleaned_plate, logged_plate):
                                        is_duplicate = True
                                        recently_logged_plates[cam_id][logged_plate] = current_time
                                        break

                            if is_duplicate:
                                continue  # Next tracker_id

                            # ── New vehicle: cooldown check then DB insert ─────
                            last_save_time = camera_save_cooldown.get(cam_id, 0.0)
                            if current_time - last_save_time < SAVE_COOLDOWN_SECONDS:
                                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [COOLDOWN] {cam_name} cooldown active "
                                      f"({SAVE_COOLDOWN_SECONDS - (current_time - last_save_time):.1f}s remaining), skipping {cleaned_plate}")
                                continue

                            v_type = classify_vehicle_type(cleaned_plate)
                            recently_logged_plates[cam_id][cleaned_plate] = current_time

                            crop_filename = f"crop_{cleaned_plate}_{timestamp_str}.jpg"
                            full_filename = f"full_{cleaned_plate}_{timestamp_str}.jpg"
                            crop_path     = os.path.join(CROPS_DIR, crop_filename)
                            full_path     = os.path.join(FULLS_DIR, full_filename)

                            _image_save_queue.put((crop_path, cropped_plate))
                            _image_save_queue.put((full_path, frame_b))

                            with db_write_lock:
                                cursor  = db_conn_persistent.cursor()
                                cursor.execute(
                                    "INSERT INTO detections (plate_number, confidence, crop_image_path, full_image_path, vehicle_type, camera_id, direction) "
                                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (cleaned_plate, round(ocr_conf, 2), crop_filename, full_filename, v_type, cam_id, direction)
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

                                db_conn_persistent.commit()

                            # ── Register in ByteTrack record table ───────────
                            tracked_by_id[cam_id][tid] = {
                                "plate":        cleaned_plate,
                                "last_seen":    current_time,
                                "vehicle_type": v_type,
                                "detection_id": detection_id,
                                "confidence":   ocr_conf,
                            }

                            camera_save_cooldown[cam_id] = current_time
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DETECTED on {cam_name}: "
                                  f"{cleaned_plate} tid={tid} (YOLO={confidence_b:.2f} OCR={ocr_conf:.2f}) "
                                  f"[best of {len(track_cands)}/{len(candidates)} candidates]")
                    else:
                        # Trigger window closed but no valid candidates were accumulated
                        plate_candidates[cam_id] = []
                        track_y_history[cam_id]  = {}
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BEST-PLATE] {cam_name}: trigger window closed with 0 valid candidates.")

                # Post-YOLO display update: re-evaluate throttle condition.
                # Since YOLO took ~200ms >> 100ms throttle interval, this always fires,
                # ensuring bounding boxes drawn above appear in the stream.
                # 改進 14 / 20: Use shared _encode_and_push() — no duplicated resize+imencode.
                if (time.time() - last_display_update.get(cam_id, 0)) >= DISPLAY_UPDATE_INTERVAL:
                    _encode_and_push(display_frame, cam_id, last_display_update)

            # 改進 7: Budget-aware sleep — only sleep remaining time in the budget window.
            # When YOLO runs (~200ms), budget is already exceeded → no sleep needed.
            # When only motion detection runs (~3ms), we sleep the remaining ~17ms.
            _loop_elapsed = time.time() - _loop_start
            _budget = 0.03 if not processed_any else 0.02
            _remaining = _budget - _loop_elapsed
            if _remaining > 0.002:   # Only sleep if > 2ms remaining (worth the syscall)
                time.sleep(_remaining)

    except KeyboardInterrupt:
        print("LPR Engine stopping...")
    finally:
        for cam_id, reader in list(active_readers.items()):
            reader.stop()
        stop_go2rtc()

if __name__ == "__main__":
    main()
