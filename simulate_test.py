import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Reconfigure output encoding for console
sys.stdout.reconfigure(encoding='utf-8')

print("===================================================")
print("     AI 車牌辨識系統 —— 離線模擬與辨識功能測試")
print("===================================================\n")

# Paths
image_dir = r"D:\AntiGravity\lpr_data\fulls"
model_path = r"D:\AntiGravity\ai camera-gate\yolov8n_lpr_openvino_model"
output_path = r"D:\AntiGravity\ai camera-gate\simulation_output.jpg"

if not os.path.exists(image_dir):
    print(f"錯誤：找不到歷史截圖資料夾 {image_dir}")
    sys.exit(1)

# Find all motorcycle or vehicle full frames
files = [f for f in os.listdir(image_dir) if f.startswith("full_") and f.endswith(".jpg")]
if not files:
    print("錯誤：lpr_data/fulls 資料夾中沒有任何存檔照片可供測試。")
    sys.exit(1)

# List some available images for testing
print("可供測試的歷史監控照片：")
test_file = None
for f in files[:10]:
    if "B6B3H5" in f or "D6K9101" in f or "BK9101" in f:
        print(f"  👉 [推薦機車照片] {f}")
        test_file = f
    else:
        print(f"  - {f}")

if not test_file:
    test_file = files[0]

print(f"\n自動選擇測試照片: {test_file}")
image_path = os.path.join(image_dir, test_file)

# Load Image
frame = cv2.imread(image_path)
if frame is None:
    print("錯誤：無法讀取測試影像檔案。")
    sys.exit(1)

# Load Models
print("\n[1/3] 正在載入 YOLOv8-LPR 車牌定位模型 (OpenVINO CPU 模式)...")
model = YOLO(model_path, task='detect')

print("[2/3] 正在載入 PaddleOCR 文字辨識模型...")
ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=False, rec_batch_num=1)

print("[3/3] 開始執行模擬推論...")
# Run YOLO
results = model(frame, conf=0.15, imgsz=640, verbose=False)

h, w, _ = frame.shape
detections_found = 0

for result in results:
    boxes = result.boxes
    print(f"-> 偵測到的目標數量: {len(boxes)}")
    for box in boxes:
        # Get coordinates
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        print(f"   [車牌框定位] 座標: [{x1}, {y1}, {x2}, {y2}] 置信度: {conf:.2f}")
        
        # Crop plate with small padding
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)
        
        cropped_plate = frame[py1:py2, px1:px2]
        if cropped_plate.size == 0:
            continue
            
        # Pad border
        cropped_padded = cv2.copyMakeBorder(
            cropped_plate,
            top=10, bottom=10, left=15, right=15,
            borderType=cv2.BORDER_REPLICATE
        )
        
        # Resize to height 64
        ph, pw, _ = cropped_padded.shape
        target_h = 64
        target_w = int(pw * (target_h / ph))
        enhanced = cv2.resize(cropped_padded, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # CLAHE Contrast Enhancement
        try:
            ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
            y_channel, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_channel = clahe.apply(y_channel)
            ycrcb = cv2.merge([y_channel, cr, cb])
            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            pass
            
        # Sharpening
        try:
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        except Exception as e:
            pass
            
        # Run PaddleOCR
        ocr_res = ocr.ocr(enhanced)
        plate_text = ""
        ocr_conf = 0.0
        
        if ocr_res and len(ocr_res) > 0 and ocr_res[0] is not None:
            res_dict = ocr_res[0]
            texts = res_dict.get('rec_texts', [])
            scores = res_dict.get('rec_scores', [])
            plate_text = "".join(texts)
            ocr_conf = max(scores) if scores else 0.0
                
        # Clean plate text
        plate_clean = "".join(c for c in plate_text if c.isalnum() or c == '-').upper()
        print(f"   [OCR 辨識結果] 車牌號碼: {plate_clean} (文字置信度: {ocr_conf:.2f})")
        
        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{plate_clean} ({ocr_conf:.2f})"
        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        detections_found += 1

if detections_found > 0:
    print(f"\n🎉 辨識測試成功！測試截圖已儲存至：\n👉 {output_path}")
    cv2.imwrite(output_path, frame)
else:
    print("\n❌ 測試照片中未偵測到任何車牌。")
