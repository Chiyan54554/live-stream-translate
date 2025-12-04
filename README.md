# 🎙️ Live Stream Translator

實時將日文直播轉錄並翻譯成繁體中文的自動化系統。

<p align="center">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 目錄

- [功能特色](#-功能特色)
- [系統架構](#-系統架構)
- [系統需求](#-系統需求)
- [快速開始](#-快速開始)
- [配置說明](#-配置說明)
- [模型比較](#-模型比較)
- [費用估算](#-費用估算)
- [常見問題](#-常見問題)
- [專案結構](#-專案結構)
- [致謝](#-致謝)

---

## ✨ 功能特色

| 功能 | 說明 |
|:----:|------|
| 🎯 | **日文語音辨識** - Kotoba-Whisper v2.2 日文優化模型 |
| 🌐 | **智能翻譯** - 本地 LLM (Ollama) 或雲端 API |
| ⚡ | **低延遲** - yt-dlp + FFmpeg 管道串流，約 4-6 秒延遲 |
| 🖥️ | **網頁介面** - 即時顯示翻譯，支援自動滾動 |
| 🐳 | **容器化** - Docker Compose 一鍵部署 |
| 🔄 | **自動重連** - 串流中斷自動恢復 |

---

## 🏗️ 系統架構

```
                           Docker Network
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Server  │───▶│  Redis   │───▶│Processor │───▶│  Ollama  │   │
│  │ (Node.js)│    │ (Pub/Sub)│    │ (Python) │    │  (LLM)   │   │
│  └────┬─────┘    └──────────┘    └──────────┘    └──────────┘   │
│       │                               │                          │
│       │ yt-dlp + FFmpeg              │ Whisper ASR              │
│       ▼                               ▼                          │
│  ┌──────────┐                   ┌──────────┐                    │
│  │ 直播平台 │                   │ 翻譯結果 │                    │
│  │Twitch/YT │                   │   JSON   │                    │
│  └──────────┘                   └────┬─────┘                    │
│                                      │                          │
└──────────────────────────────────────┼──────────────────────────┘
                                       │ WebSocket
                                       ▼
                                 ┌──────────┐
                                 │  瀏覽器  │
                                 │  :8080   │
                                 └──────────┘
```

---

## 📦 系統需求

### 硬體

| 項目 | 最低需求 | 推薦配置 |
|------|:--------:|:--------:|
| GPU | NVIDIA 8GB VRAM | NVIDIA 12GB+ VRAM |
| RAM | 8GB | 16GB |
| 硬碟 | 20GB | 50GB |

### 軟體

- Docker Desktop 4.0+
- NVIDIA Container Toolkit
- NVIDIA Driver 535+

### VRAM 使用量

| ASR 模型 | 翻譯引擎 | VRAM |
|----------|----------|:----:|
| `medium` | Google Translate | ~5GB |
| `large-v3-turbo` | Google Translate | ~6GB |
| `kotoba-v2.2` | Google Translate | ~10GB |
| `kotoba-v2.2` | Ollama qwen3:8b | ~14GB |

---

## 🚀 快速開始

### 1️⃣ 複製專案

```bash
git clone https://github.com/chiyan/live-stream-translate.git
cd live-stream-translate
```

### 2️⃣ 設定直播 URL

編輯 `server/server.js`：

```javascript
const LIVE_PAGE_URL = 'https://www.twitch.tv/your-channel';
```

### 3️⃣ 啟動服務

```bash
docker-compose up --build
```

> ⏳ 首次啟動需下載模型（約 10-15GB），請耐心等待

### 4️⃣ 開啟網頁

```
http://localhost:8080
```

---

## ⚙️ 配置說明

### ASR 模型

編輯 `docker-compose.yml`：

```yaml
environment:
  # === 日文優化（推薦）===
  ASR_MODEL_NAME: kotoba-tech/kotoba-whisper-v2.2
  
  # === 標準 Whisper ===
  # ASR_MODEL_NAME: large-v3
  # ASR_MODEL_NAME: large-v3-turbo    # 較快
  # ASR_MODEL_NAME: medium            # 省 VRAM
```

### 翻譯引擎

| 引擎 | 設定 | 費用 |
|------|------|:----:|
| Ollama (預設) | `LLM_MODEL: qwen3:8b` | 免費 |
| Google Translate | 修改 `translator.py` | 免費 |
| OpenAI | `OPENAI_API_KEY: xxx` | ~$0.20/2hr |

### 緩衝設定

編輯 `processor/config.py`：

```python
BUFFER_DURATION_S = 4.0     # 音訊緩衝（秒）
MIN_PUBLISH_INTERVAL = 0.8  # 發布間隔（秒）
```

---

## 📊 模型比較

### ASR 模型

| 模型 | 日文準確度 | 速度 | VRAM |
|------|:----------:|:----:|:----:|
| kotoba-v2.2 | ⭐⭐⭐⭐⭐ | 中 | 10GB |
| large-v3 | ⭐⭐⭐⭐ | 中 | 10GB |
| large-v3-turbo | ⭐⭐⭐⭐ | 快 | 6GB |
| medium | ⭐⭐⭐ | 快 | 5GB |

### 翻譯引擎

| 引擎 | 品質 | 速度 | 費用 |
|------|:----:|:----:|:----:|
| Ollama qwen3:8b | ⭐⭐⭐⭐ | 中 | 免費 |
| Google Translate | ⭐⭐⭐ | 快 | 免費 |
| GPT-4o-mini | ⭐⭐⭐⭐⭐ | 快 | $0.20/2hr |

---

## 💰 費用估算

### 2 小時直播

| 配置 | 費用 |
|------|:----:|
| 本地 Whisper + Ollama | **$0** |
| 本地 Whisper + Google | **$0** |
| 本地 Whisper + GPT-4o-mini | ~$0.20 |
| Deepgram + GPT-4o-mini | ~$0.70 |

---

## ❓ 常見問題

<details>
<summary><b>模型下載很慢？</b></summary>

使用 HuggingFace 鏡像：

```yaml
environment:
  HF_ENDPOINT: https://hf-mirror.com
```
</details>

<details>
<summary><b>CUDA out of memory？</b></summary>

1. 使用較小模型：`ASR_MODEL_NAME: medium`
2. 關閉 Ollama，改用 Google Translate
3. 減少 `BUFFER_DURATION_S`
</details>

<details>
<summary><b>翻譯有重複內容？</b></summary>

調整 `processor/config.py`：

```python
SIMILARITY_THRESHOLD = 0.75  # 提高此值
```
</details>

<details>
<summary><b>yt-dlp 無法擷取串流？</b></summary>

1. 確認直播正在進行中
2. 更新 yt-dlp：重建 Docker 映像
3. 檢查網路連線
</details>

---

## 📁 專案結構

```
live-stream-translate/
├── 📄 docker-compose.yml     # 服務編排
├── 📄 Dockerfile.server      # Node.js 映像
├── 📄 client.html            # 網頁前端
│
├── 📂 server/
│   ├── server.js             # 串流處理 + WebSocket
│   └── package.json
│
├── 📂 processor/
│   ├── Dockerfile.processor  # Python 映像
│   ├── main.py               # 主程式入口
│   ├── asr.py                # 語音辨識
│   ├── translator.py         # 翻譯模組
│   ├── text_utils.py         # 文字處理
│   └── config.py             # 配置檔
│
└── 📂 ollama/
    ├── Dockerfile.ollama     # Ollama 映像
    └── entrypoint.sh         # 啟動腳本
```

---

## 📝 更新日誌

### v1.0.0 (2024-01)
- ✅ 初始版本
- ✅ 支援 Twitch / YouTube
- ✅ Kotoba-Whisper v2.2
- ✅ Ollama 本地翻譯

---

## 📄 授權

[MIT License](LICENSE)

---

## 🙏 致謝

| 專案 | 用途 |
|------|------|
| [OpenAI Whisper](https://github.com/openai/whisper) | 語音辨識基礎 |
| [Kotoba-Whisper](https://huggingface.co/kotoba-tech) | 日文優化模型 |
| [stable-ts](https://github.com/jianfch/stable-ts) | 時間戳優化 |
| [Ollama](https://ollama.ai) | 本地 LLM |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | 串流擷取 |

---

<p align="center">
  Made with ❤️ for VTuber fans
</p>
