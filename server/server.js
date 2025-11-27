const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
const Redis = require('ioredis'); 

// --- 配置參數 ---
const WSS_PORT = 8080;
// ⚠️ 這是 Twitch 或 YouTube 的網頁 URL，供 yt-dlp 解析
const LIVE_PAGE_URL = 'https://www.twitch.tv/videos/2626749881'; 

// Redis 配置 (從環境變量讀取，供 Docker 使用)
const REDIS_HOST = process.env.REDIS_HOST || 'localhost'; 
const REDIS_PORT = parseInt(process.env.REDIS_PORT) || 6379; 

const AUDIO_CHANNEL = "audio_feed";           // 📢 Node.js -> Python
const TRANSLATION_CHANNEL = "translation_feed"; // 👂 Python -> Node.js

let ffmpegProcess = null;
let publisher; 
let subscriber; 
let wss; 

// [ WebSocket 啟動和連線邏輯，保持不變 ]
const server = http.createServer((req, res) => {
    // 服務 client.html
    if (req.url === '/') {
        fs.readFile(path.join(__dirname, '../client.html'), (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading client.html');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else {
        res.writeHead(404);
        res.end();
    }
});

wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('Client connected.');
    ws.on('close', () => console.log('Client disconnected.'));
});

server.listen(WSS_PORT, () => {
    console.log(`Node.js WebSocket Server 启动在 ws://localhost:${WSS_PORT}`);
    startMainFlow();
});


// 1. 初始化 Redis 客戶端並訂閱翻譯結果
function initializeRedisClients() {
    publisher = new Redis({ host: REDIS_HOST, port: REDIS_PORT });
    subscriber = new Redis({ host: REDIS_HOST, port: REDIS_PORT });

    publisher.on('error', (err) => { console.error('致命錯誤：Redis Publisher 連線錯誤:', err); });
    subscriber.on('error', (err) => { console.error('致命錯誤：Redis Subscriber 連線錯誤:', err); });
    publisher.on('connect', () => { console.log('Redis Publisher 連線成功。'); });
    subscriber.on('connect', () => { console.log('Redis Subscriber 連線成功。'); });

    // 訂閱翻譯結果頻道 (來自 Python)
    subscriber.subscribe(TRANSLATION_CHANNEL, (err, count) => {
        if (err) {
            console.error('致命錯誤：Redis 訂閱翻譯頻道失敗:', err);
            return;
        }
        console.log(`Node.js 成功訂閱 Redis 頻道: ${TRANSLATION_CHANNEL} (${count} 個頻道)。`);
    });

    // 處理接收到的 Redis 消息 (翻譯結果)
    subscriber.on('message', (channel, message) => {
        if (channel === TRANSLATION_CHANNEL) {
            try {
                // Redis 傳輸保證數據清潔，直接廣播給所有 WebSocket 客戶端
                JSON.parse(message); // 快速驗證
                wss.clients.forEach(client => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(message); // 發送原始 JSON 字符串
                    }
                });
            } catch (error) {
                console.error('致命錯誤：無法解析 Redis 接收到的 JSON 數據:', error.message);
            }
        }
    });
}

// 2. 啟動 FFmpeg 處理器，並將輸出發佈到 Redis
function startFFmpegProcessor(streamUrl) {
    console.log('--- 啟動 FFmpeg 進程抓取直播流 ---');
    
    // FFmpeg 參數 (使用 16kHz 採樣率，與 Python 保持一致)
    const ffmpegArgs = [
        '-re', 
        '-nostdin', 
        '-analyzeduration', '10000000', 
        '-probesize', '10000000',
        '-loglevel', 'error', 
        '-i', streamUrl,
        '-ac', '1', 
        '-ar', '16000', // 16kHz
        '-acodec', 'pcm_s16le', // 16-bit PCM
        '-f', 's16le', 
        'pipe:1' // 輸出到 stdout
    ];

    ffmpegProcess = spawn('ffmpeg', ffmpegArgs, {
        stdio: ['ignore', 'pipe', process.stderr] // FFmpeg 的 stderr 直接管到 Node.js 的 stderr
    });
    
    console.log(`--- FFmpeg 輸出管道 -> Node.js -> Redis 頻道: ${AUDIO_CHANNEL} ---`);

    // 關鍵：將 FFmpeg 的 stdout 數據塊發佈到 Redis 
    ffmpegProcess.stdout.on('data', (audioChunk) => {
        // 將 Buffer 轉換為 Base64 字符串發佈，以便 Python 接收
        const base64Audio = audioChunk.toString('base64');
        
        publisher.publish(AUDIO_CHANNEL, base64Audio).catch(err => {
            console.error('致命錯誤：發佈音頻數據到 Redis 失敗:', err);
        });
    });

    ffmpegProcess.on('error', (err) => console.error('FFmpeg 啟動失敗:', err));
    ffmpegProcess.on('close', (code) => {
        console.log(`FFmpeg 进程退出, Code: ${code}.`);
    });
}

// 3. 獲取直播 URL (yt-dlp 邏輯保持不變)
function getStreamUrl(callback) {
    console.log(`--- 正在使用 yt-dlp 解析直播串流 URL: ${LIVE_PAGE_URL} ---`);
    const YTDLP_EXEC_PATH = 'yt-dlp'; 
    const ytdlp = spawn(YTDLP_EXEC_PATH, ['-f', 'bestaudio', '--get-url', LIVE_PAGE_URL]);
    let streamUrl = '';

    ytdlp.stdout.on('data', (data) => {
        streamUrl += data.toString().trim();
    });
    
    ytdlp.stderr.on('data', (data) => {
        console.error(`[yt-dlp 调试/警告]: ${data.toString().trim()}`);
    });

    ytdlp.on('close', (code) => {
        if (code === 0 && streamUrl) {
            console.log('✅ yt-dlp 解析成功，獲取到串流 URL。');
            callback(streamUrl.split('\n')[0]);
        } else {
            console.error('❌ yt-dlp 解析失敗，無法獲取串流 URL。');
            callback(null);
        }
    });
}

function startMainFlow() {
    initializeRedisClients();
    getStreamUrl((streamUrl) => {
        if (!streamUrl) return;
        startFFmpegProcessor(streamUrl); 
    });
}