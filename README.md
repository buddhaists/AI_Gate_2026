# AI 智慧車牌辨識與監控系統 —— 安裝與使用手冊

本手冊提供將 **AI 智慧學校大門車牌監控系統 (YOLOv8-LPR + PaddleOCRv6)** 部署、配置、運作於新電腦的完整指南。

---

## 1. 系統概述
本系統是一款專為學校大門設計的輕量化、高靈敏度車牌辨識與監控系統：
* **後端架構**：Python (HTTP Server) + RTSP 即時影像解碼執行緒 + 運動偵測分析執行緒。
* **AI 辨識引擎**：
  * **YOLOv8-LPR**：進行即時車牌定位（經 OpenVINO CPU 向量加速，單次推論僅需 27ms）。
  * **PaddleOCRv6**：對車牌影像進行精準文字辨識。
* **動態追蹤與過濾**：
  * **Supervision ByteTrack**：採用 Kalman 濾波算法對車牌偵測框進行連續格訊號追蹤，取代傳統 IoU 去重，大幅提升移動中車輛與多車並行時的辨識精準度與防重複儲存能力。
  * **Supervision PolygonZone**：閘門偵測區多邊形過濾，保證僅在指定的車道/大門區內進行車牌辨識，完全過濾框外的背景車流。
* **前端介面**：採用現代 Glassmorphism (毛玻璃) 風格網頁儀表板，支援 RWD 響應式佈局（電腦、平板、手機皆可流暢瀏覽），並具備網頁端多邊形區域編輯器，免重啟熱更新。

---

## 2. 硬體與作業系統需求

### 💻 建議硬體配置
* **處理器 (CPU)**：Intel Core i5/i7 第 10 代以上（具備強健的 AVX2 / AVX-512 指令集，最能發揮 OpenVINO CPU 加速效能，且運作極度省電穩定）。
* **記憶體 (RAM)**：8 GB 以上。
* **儲存空間 (ROM)**：預留至少 10 GB 空間用於存放車流照片日誌與 SQLite 資料庫（系統會自動依設定天數清理過期照片）。

### 💿 作業系統
* Windows 10 / Windows 11 64-bit。

### 🌐 網路環境
* 伺服器主機與 IP 攝影機（或 NVR 錄影機）必須處於同一個區域網路網段內（例如 `192.168.1.x`）。

---

## 3. 環境與系統安裝步驟

### 🔹 第一步：安裝 Python 3.10
1. 下載並安裝 [Python 3.10.11 (64-bit)](https://www.python.org/downloads/release/python-31011/)。
2. ⚠️ **安裝時的重要勾選**：請務必在安裝介面第一頁勾選 **「Add Python 3.10 to PATH」**（將 Python 加入系統環境變數）。
3. 安裝完成後，開啟命令提示字元 (CMD) 輸入 `python --version`，確認有正確顯示 `Python 3.10.11`。

### 🔹 第二步：複製專案專屬檔案
將備份目錄中的所有檔案複製到新電腦的硬碟中（例如放置在：`D:\AntiGravity\ai camera-gate`）。
專案目錄中必須包含以下核心結構：
```
ai camera-gate/
├── public/                 # 前端網頁靜態資源 (HTML/CSS/JS)
├── yolov8n_lpr_openvino_model/  # OpenVINO 格式的 YOLO 辨識模型
├── lpr_engine.py           # 後端 Python 主程式
├── run_server.bat          # 一鍵啟動批次檔
├── requirements.txt        # Python 依賴清單
├── ffmpeg.exe              # (⚠️ 外部二進位) 影像解碼執行檔
├── go2rtc.exe              # (⚠️ 外部二進位) RTSP 串流代理執行檔
└── go2rtc.yaml             # RTSP 代理設定檔
```
> [!IMPORTANT]
> `ffmpeg.exe` (約 136MB) 與 `go2rtc.exe` (約 19MB) 因檔案過大未上傳至 GitHub，請務必直接從您的 **本地備份資料夾** (`智慧大門_網頁備份026.zip`) 複製這兩個檔案到新電腦的專案根目錄。

### 🔹 第三步：安裝 Python 依賴庫
開啟 CMD 視窗，切換至專案根目錄，並執行一鍵安裝指令：
```cmd
cd /d "D:\AntiGravity\ai camera-gate"
pip install -r requirements.txt
```
為確保 PaddlePaddle 影像運算庫最佳的 CPU 穩定度，建議安裝 CPU 專用版：
```cmd
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

---

## 4. 系統設定與配置

本專案將憑證與設定完全隔離，請依據新電腦的環境修改以下兩個檔案：

### 📝 1. `config.json` (系統設定)
用文字編輯器（如記事本）打開根目錄下的 `config.json`：
```json
{
  "web_password": "您的網頁密碼", 
  "telegram_token": "您的Telegram機器人Token",
  "telegram_chat_id": "您的Telegram聊天室ID",
  "cameras": [
    {
      "id": 1,
      "name": "學校大門",
      "url": "rtsp://帳號:密碼@攝影機或NVR_IP:554/串流路徑"
    },
    {
      "id": 2,
      "name": "學校大門002",
      "url": "rtsp://帳號:密碼@攝影機或NVR_IP:554/串流路徑"
    }
  ]
}
```

### 📝 2. `go2rtc.yaml` (串流代理設定)
開啟根目錄下的 `go2rtc.yaml`：
```yaml
streams:
  # 對應 config.json 中的攝影機，讓前端能以網頁無插件技術即時拉流
  cam_1: "rtsp://帳號:密碼@攝影機_IP:554/串流路徑"
  cam_2: "rtsp://帳號:密碼@攝影機_IP:554/串流路徑"
```

---

## 5. 啟動與運行系統

雙擊專案目錄下的 **`run_server.bat`** 批次檔，系統便會自動在背景啟動所有服務。

### 🔍 確認啟動正常
當 CMD 視窗顯示以下日誌時，代表啟動成功：
1. `Using OpenVINO LATENCY mode for batch=1 inference on (CPU)...` (OpenVINO CPU 加速啟動)
2. `[SYSTEM] OpenVINO initialized on CPU successfully.`
3. `[SYSTEM] ByteTrack initialized for camera X...` (ByteTrack 模組啟動)
4. `AI LPR Engine started successfully. Monitoring stream...` (監控主引擎上線)
5. `Web Dashboard Server running at http://localhost:8081` (網頁伺服器開啟)
6. `RTSP proxy connection established.` (影像串流連線成功)

---

## 6. 前端網頁操作說明

### 🌐 瀏覽器連線網址
* **主機本機存取**：在伺服器上打開瀏覽器輸入：[http://localhost:8081/](http://localhost:8081/)
* **區域網路內其他電腦/手機存取**：
  * **主機名虛擬網址**：`http://FSPSMIS2021.local:8081/`（FSPSMIS2021 為您目前的電腦名稱）
  * **IP 直連網址**：`http://192.168.1.93:8081/`（請替換為新電腦的實體 IP）

### 🔑 登入驗證
* **預設登入密碼**為：`3762828`（可在 `config.json` 或設定分頁中隨時更改）。

### 📺 網頁分頁功能
1. **智慧儀表板 (Dashboard)**：
   - 顯示今日偵測車流總數、系統 CPU/記憶體狀態。
   - 即時串流監視窗：點擊右上角綠色圓點可看即時影像。
   - 側邊欄：顯示最新辨識出的 5 筆車牌，包含車牌大圖截圖。
2. **歷史紀錄查詢 (History)**：
   - 可依據「車牌號碼」進行模糊查詢（如輸入 `BDP` 即可查出 `BDP3832` 等車輛）。
   - 顯示所有通過時間、相機來源，並提供辨識時的車牌大圖下載。
3. **偵測區域設定 (Detection Zones)**：
   - 可選擇不同相機，拖曳畫面上的青色多邊形圓點，微調 `sv.PolygonZone` 閘門偵測區。
   - 按下「儲存區域」後，全新座標將立即寫入後端 `D:\AntiGravity\lpr_data\zone_config.json` 並即時熱更新生效，無須重啟引擎。
4. **系統維運設定 (Settings)**：
   - **攝影機管理**：新增/刪除/修改相機 RTSP 串流網址。
   - **網頁帳密設定**：線上即時修改網頁密碼。
   - **Telegram 推播**：可開啟/關閉通知，或修改通知 Token。
   - **維運設定**：設定車流資料與截圖照片的自動清理天數（預設為 60 天），保護硬碟不爆滿。

---

## 7. 最佳化維護與故障排除

### 🔌 高效能與執行緒控制
本系統出廠前已完成高度效能優化，主要控制變數位於 `lpr_engine.py` 頂部：
* `KMP_BLOCKTIME=0`：確保 OpenVINO CPU 推論核心在沒有車輛經過時，**立刻進入休眠**，不會空轉。
* `TBB_MAX_ALLOWED_NUM_THREADS=1`：限制背景執行緒池，避免多執行緒競爭造成 CPU 飆高。
* 若在新電腦上發現 CPU 負載有異常，請確認上述環境變數是否有正常被讀取。

### 🍂 運動偵測與閘門區域過濾調校
新版系統結合了「運動偵測」與「多邊形偵測區域過濾」：
1. **過濾無效移動**：若攝影機拍攝範圍較大，包含路樹晃動、校外道路車流等。請在「偵測區域設定」面板中，將多邊形範圍限縮在大門口區域。
2. **只保留框內觸發**：當校外道路有車通過時，雖然會觸發運動偵測，但因為偵測框位於 `PolygonZone` 遮罩之外，系統會自動在日誌中顯示 `[ZONE] 00X大門: X/X detections rejected (outside gate zone)` 並將其捨棄，不調用 LPR 辨識，從根本上杜絕外部無效警報，並釋放 CPU 運算資源。
