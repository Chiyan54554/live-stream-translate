# Live Stream Real-time Translation System (直播實時翻譯系統)

這是一個基於 Docker 的實時直播翻譯系統，能夠抓取直播音訊（如 Twitch），使用 OpenAI Whisper 進行語音轉文字（ASR），並透過 Google Translate 進行翻譯，最後將字幕實時推送到 Web 客戶端顯示。

![License](https://img.shields.io/badge/license-ISC-blue.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Node.js](https://img.shields.io/badge/node.js-6DA55F?style=flat&logo=node.js&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)

## ✨ 功能特色

*   **實時音訊擷取**：使用 `yt-dlp` 和 `ffmpeg` 從直播源（如 Twitch）提取音訊流。
*   **AI 語音轉錄**：整合 **OpenAI Whisper** 模型（支援 GPU 加速），提供高精度的日語（或其他語言）轉錄。
*   **即時翻譯**：使用 `deep-translator` (Google Translate) 將轉錄文本翻譯成繁體中文。
*   **智能過濾**：內建過濾機制，自動去除 Whisper 常見的幻覺文本（如「ご視聴ありがとうございました」）和非語言噪音。
*   **WebSocket 推送**：後端透過 WebSocket 將翻譯結果即時推送到前端。
*   **現代化 UI**：響應式 Web 介面，支援深色/淺色模式，提供舒適的觀看體驗。
*   **Docker 化部署**：一鍵啟動所有服務（Redis, Node.js Server, Python Processor）。

## 🏗️ 系統架構

系統由三個主要 Docker 容器組成，透過 Redis 進行通訊：

```mermaid
graph TD
    Live[直播源 (Twitch)] -->|yt-dlp/ffmpeg| Node[Node.js Server]
    Node -->|音訊數據 (Pub)| Redis[(Redis Message Broker)]
    Redis -->|音訊數據 (Sub)| Python[Python Processor]
    Python -->|Whisper ASR + 翻譯| Python
    Python -->|翻譯結果 (Pub)| Redis
    Redis -->|翻譯結果 (Sub)| Node
    Node -->|WebSocket| Client[Web Client (Browser)]
```

1.  **Node.js Server**: 負責抓取直播流，將音訊切片發送至 Redis，並作為 WebSocket 伺服器向前端廣播翻譯結果。
2.  **Redis**: 作為訊息佇列（Message Broker），處理音訊流和翻譯結果的傳遞。
3.  **Python Processor**: 訂閱音訊流，執行 Whisper 模型進行轉錄和翻譯，並將結果回傳。

## 🚀 快速開始

### 前置需求

*   **Docker** & **Docker Compose**
*   **NVIDIA GPU** (強烈建議): 需安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 以支援 Whisper GPU 加速。
    *   *如果沒有 GPU，需修改 `docker-compose.yml` 和程式碼以使用 CPU 模式（速度會較慢）。*

### 安裝與執行

1.  **複製專案**
    ```bash
    git clone https://github.com/YourUsername/live-stream-translate.git
    cd live-stream-translate
    ```

2.  **啟動服務**
    使用 Docker Compose 建置並啟動所有服務：
    ```bash
    docker-compose up --build
    ```
    *首次啟動需要下載 Whisper 模型和 Docker 映像檔，請耐心等待。*

3.  **開啟客戶端**
    直接在瀏覽器中打開專案根目錄下的 `client.html` 文件，或將其部署到網頁伺服器。
    *   預設連接 WebSocket 地址: `ws://localhost:8080`

## ⚙️ 配置說明

### 修改直播源
目前直播 URL 設定在 `server/server.js` 中。若要更改目標直播頻道：

1.  打開 `server/server.js`
2.  修改 `LIVE_PAGE_URL` 變數：
    ```javascript
    const LIVE_PAGE_URL = 'https://www.twitch.tv/your_target_channel';
    ```
3.  重啟 Node.js 容器：
    ```bash
    docker-compose restart server
    ```

### 修改 Whisper 模型
可以在 `docker-compose.yml` 中調整使用的 Whisper 模型大小（預設為 `medium`）：

```yaml
environment:
  ASR_MODEL_NAME: large-v2  # 可選: tiny, base, small, medium, large, large-v2
```
*注意：模型越大，準確度越高，但對 VRAM 的需求也越高。*

## 🛠️ 技術棧

*   **Frontend**: HTML5, CSS3 (Responsive), JavaScript (WebSocket)
*   **Backend**: Node.js, Express, `fluent-ffmpeg`, `yt-dlp`
*   **AI/Processing**: Python 3, OpenAI Whisper, PyTorch, Deep Translator
*   **Infrastructure**: Docker, Redis

## 📝 注意事項

*   **GPU 支援**: 確保您的 Docker Host 已正確配置 NVIDIA Runtime，否則 Python 容器可能無法使用 GPU。
*   **延遲**: 由於直播流緩衝、音訊切片（預設 128ms）和模型推論時間，翻譯字幕會有幾秒鐘的延遲是正常的。
*   **Twitch 廣告**: 直播中的廣告可能會干擾音訊抓取，建議使用無廣告的源或自行處理廣告片段。

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request 來改進這個專案！

## 📄 授權

ISC License
