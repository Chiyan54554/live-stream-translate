# 🎙️ 直播即時翻譯系統

將日文直播即時轉錄並翻譯成繁體中文的自動化系統。

## ✨ 功能特色

- 🎯 **即時語音辨識**：使用 Kotoba-Whisper v2.2（日文優化）或 Whisper large-v3
- 🌐 **智能翻譯**：支援本地 LLM (Ollama) 或 Google Translate
- ⚡ **低延遲串流**：yt-dlp + FFmpeg 管道處理
- 🖥️ **網頁介面**：即時顯示翻譯結果
- 🐳 **Docker 部署**：一鍵啟動所有服務

## 🏗️ 系統架構

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   直播平台   │───▶│  Node.js    │───▶│   Redis     │───▶│   Python    │
│ Twitch/YT   │    │ yt-dlp+FFmpeg│    │  Pub/Sub    │    │  Processor  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                   ┌─────────────┐    ┌─────────────┐           │
                   │   瀏覽器    │◀───│  WebSocket  │◀──────────┘
                   │  client.html │    │   Server    │
                   └─────────────┘    └─────────────┘
```

## 📦 系統需求

- **GPU**：NVIDIA GPU（建議 8GB+ VRAM）
- **Docker**：Docker Desktop with NVIDIA Container Toolkit
- **作業系統**：Windows 10/11、Linux、macOS

### VRAM 需求

| 配置 | ASR 模型 | 翻譯引擎 | 總 VRAM |
|------|----------|----------|---------|
| 最低 | medium | Google Translate | ~5GB |
| 推薦 | kotoba-v2.2 | Ollama qwen3:8b | ~12GB |
| 高階 | large-v3 | Ollama qwen3:8b | ~16GB |

## 🚀 快速開始

### 1. 複製專案

```bash
git clone https://github.com/your-repo/live-stream-translate.git
cd live-stream-translate
```

### 2. 設定直播 URL

編輯 `server/server.js`：

```javascript
const LIVE_PAGE_URL = 'https://www.twitch.tv/your-channel';
```

### 3. 啟動服務

```bash
docker-compose up --build
```

首次啟動會下載模型（約 10-15GB），請耐心等待。

### 4. 開啟網頁

瀏覽器開啟：http://localhost:8080

## ⚙️ 配置說明

### ASR 模型選擇

編輯 `docker-compose.yml`：

```yaml
environment:
  # 日文優化（推薦）
  ASR_MODEL_NAME: kotoba-tech/kotoba-whisper-v2.2
  
  # 標準 Whisper（備選）
  # ASR_MODEL_NAME: large-v3
  # ASR_MODEL_NAME: large-v3-turbo  # 較快
  # ASR_MODEL_NAME: medium          # 省 VRAM
```

### 翻譯引擎選擇

目前支援：
- **Ollama (qwen3:8b)**：本地 LLM，免費，品質佳
- **Google Translate**：免費，速度快

## 📁 專案結構

```
live-stream-translate/
├── docker-compose.yml      # Docker 服務編排
├── Dockerfile.server       # Node.js 服務映像
├── client.html             # 網頁前端
├── server/
│   ├── server.js           # Node.js 主程式
│   └── package.json
├── processor/
│   ├── Dockerfile.processor
│   ├── main.py             # Python 主程式
│   ├── asr.py              # 語音辨識模組
│   ├── translator.py       # 翻譯模組
│   ├── text_utils.py       # 文字處理
│   └── config.py           # 配置檔
└── ollama/
    └── Dockerfile.ollama   # Ollama 服務映像
```

## 🔧 進階設定

### 調整緩衝時長

編輯 `processor/config.py`：

```python
BUFFER_DURATION_S = 4.0     # 音訊緩衝（秒）
MIN_PUBLISH_INTERVAL = 0.8  # 最小發布間隔（秒）
```

### 使用 OpenAI API

1. 建立 `.env` 檔案：

```env
OPENAI_API_KEY=sk-your-api-key
```

2. 修改 `docker-compose.yml`：

```yaml
environment:
  OPENAI_API_KEY: ${OPENAI_API_KEY}
  OPENAI_MODEL: gpt-4o-mini
```

## 📊 費用估算

### 2 小時直播

| 配置 | 費用 |
|------|------|
| 本地 Whisper + Ollama | **免費** |
| 本地 Whisper + Google Translate | **免費** |
| 本地 Whisper + GPT-4o-mini | ~$0.20 |
| OpenAI Whisper + GPT-4o-mini | ~$0.90 |

## 🐛 常見問題

### Q: 模型下載很慢？

使用 HuggingFace 鏡像：

```yaml
environment:
  HF_ENDPOINT: https://hf-mirror.com
```

### Q: CUDA out of memory？

1. 使用較小的模型：`ASR_MODEL_NAME: medium`
2. 關閉 Ollama，改用 Google Translate

### Q: 翻譯有重複內容？

系統已內建去重機制，如仍有問題可調整：

```python
SIMILARITY_THRESHOLD = 0.7  # 提高此值
```

### Q: yt-dlp 無法擷取串流？

1. 更新 yt-dlp：`pip install -U yt-dlp`
2. 確認直播連結正確且正在直播中

## 📝 更新日誌

### v1.0.0
- 初始版本
- 支援 Twitch / YouTube 直播
- Kotoba-Whisper v2.2 日文優化
- Ollama 本地 LLM 翻譯

## 📄 授權

MIT License

## 🙏 致謝

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech)
- [stable-ts](https://github.com/jianfch/stable-ts)
- [Ollama](https://ollama.ai)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
