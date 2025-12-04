const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
// 引入 Redis
const Redis = require('ioredis'); 

// --- 配置參數 (預先計算的常數) ---
const WSS_PORT = 8080; 
const LIVE_PAGE_URL = 'https://www.twitch.tv/sakuramimiru'; // 直播頁面 URL

// Redis 配置
const REDIS_HOST = process.env.REDIS_HOST || 'localhost'; 
const REDIS_PORT = parseInt(process.env.REDIS_PORT) || 6379; 

const AUDIO_CHANNEL = "audio_feed";           // Node.js -> Python (發佈音頻)
const TRANSLATION_CHANNEL = "translation_feed"; // Python -> Node.js (訂閱翻譯)

const SAMPLE_RATE = 16000;
const BYTES_PER_SAMPLE = 2; // 16-bit PCM = 2 Bytes

// 定義每個音訊塊的時長 (決定 Redis 發佈頻率)
// 🎯 配合 Python 端 2 秒緩衝，縮短發送間隔
const CHUNK_DURATION_S = 0.25; // 🎯 每 0.25 秒發送一次（加快響應）

// 🎯 預先計算的常數 (避免運行時計算)
const TARGET_CHUNK_SIZE_BYTES = 8000; // Math.ceil(0.25 * 16000 * 2) = 8000

// 🎯 預先建立的 Redis 連線選項 (避免每次重建物件)
const REDIS_OPTIONS = Object.freeze({
    host: REDIS_HOST,
    port: REDIS_PORT,
    retryStrategy: (times) => Math.min(times * 100, 3000),
    maxRetriesPerRequest: 3,
    enableReadyCheck: false,
    lazyConnect: false,
});

// 🎯 預先建立的 yt-dlp 基礎參數 (凍結陣列防止意外修改)
const YTDLP_BASE_ARGS = Object.freeze([
    '-f', 'bestaudio/best',
    '--no-warnings',
    '--force-ipv4',
    '--no-check-certificate',
    '--no-playlist',
    '-o', '-',
]);

// 🎯 預先建立的 FFmpeg 參數
const FFMPEG_ARGS = Object.freeze([
    '-fflags', '+nobuffer+flush_packets',
    '-flags', 'low_delay',
    '-i', 'pipe:0',
    '-acodec', 'pcm_s16le',
    '-ar', '16000',
    '-ac', '1',
    '-f', 's16le',
    '-flush_packets', '1',
    'pipe:1'
]);

// 🎯 預編譯的平台檢測 (O(1) Set 查找)
const YOUTUBE_DOMAINS = new Set(['youtube.com', 'youtu.be']);
const TWITCH_DOMAINS = new Set(['twitch.tv']);

// 🎯 預先讀取 client.html (避免每次請求都讀檔)
const CLIENT_HTML_PATH = path.join(__dirname, '../client.html');
let cachedClientHtml = null;
fs.readFile(CLIENT_HTML_PATH, (err, data) => {
    if (!err) {
        cachedClientHtml = data;
        console.log('✅ client.html 已預先快取');
    }
});

// 🎯 平台檢測函數 (使用 Set 的 O(1) 查找)
const isYouTube = YOUTUBE_DOMAINS.has('youtube.com') || YOUTUBE_DOMAINS.has('youtu.be') 
    ? LIVE_PAGE_URL.includes('youtube.com') || LIVE_PAGE_URL.includes('youtu.be')
    : false;
const isTwitch = LIVE_PAGE_URL.includes('twitch.tv');

let ffmpegProcess = null;
let publisher; // Redis publisher client
let subscriber; // Redis subscriber client
let wss; 

// 🎯 預建立的 HTTP 回應標頭 (避免重複建立物件)
const HTML_HEADERS = Object.freeze({ 'Content-Type': 'text/html' });

// [ WebSocket 啟動和連線邏輯 ]
const server = http.createServer((req, res) => {
    // 🎯 使用快取的 client.html
    if (req.url === '/') {
        if (cachedClientHtml) {
            res.writeHead(200, HTML_HEADERS);
            res.end(cachedClientHtml);
        } else {
            // 快取未就緒時的備援方案
            fs.readFile(CLIENT_HTML_PATH, (err, data) => {
                if (err) {
                    res.writeHead(500);
                    res.end('Error loading client.html');
                    return;
                }
                cachedClientHtml = data; // 同時更新快取
                res.writeHead(200, HTML_HEADERS);
                res.end(data);
            });
        }
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
    // 🎯 使用預建立的 Redis 選項
    publisher = new Redis(REDIS_OPTIONS);
    subscriber = new Redis(REDIS_OPTIONS);

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

    // 🎯 處理接收到的 Redis 消息 - 優化版本
    subscriber.on('message', (channel, message) => {
        if (channel !== TRANSLATION_CHANNEL) return; // 🎯 早期返回
        
        // 🎯 快速 JSON 驗證 (只檢查首尾字元)
        const len = message.length;
        if (len < 2) return;
        const firstChar = message.charCodeAt(0);
        const lastChar = message.charCodeAt(len - 1);
        // 123 = '{', 125 = '}', 91 = '[', 93 = ']'
        const isLikelyJson = (firstChar === 123 && lastChar === 125) || 
                            (firstChar === 91 && lastChar === 93);
        
        if (!isLikelyJson) return;
        
        // 🎯 使用 for...of 迭代器 (比 forEach 更快)
        const clients = wss.clients;
        for (const client of clients) {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        }
    });
}

// 2. 啟動串流處理 (yt-dlp -> Pipe -> FFmpeg -> Redis)
function startStreamProcessing(publisher) {
    console.log(`--- 正在使用 yt-dlp 啟動串流處理: ${LIVE_PAGE_URL} ---`);
    const YTDLP_EXEC_PATH = 'yt-dlp';
    const FFMPEG_EXEC_PATH = 'ffmpeg';
    
    // 🎯 使用預建立的參數陣列，只在需要時添加平台特定參數
    const ytdlpArgs = [...YTDLP_BASE_ARGS]; // 淺拷貝預建立的陣列
    
    // 🎯 平台特定參數 (使用預先計算的布林值)
    if (isYouTube) {
        ytdlpArgs.push('--live-from-start', '--extractor-args', 'youtube:skip=dash');
    } else if (isTwitch) {
        ytdlpArgs.push('--referer', 'https://www.twitch.tv/');
    }
    
    ytdlpArgs.push(LIVE_PAGE_URL);
    
    const ytdlpProcess = spawn(YTDLP_EXEC_PATH, ytdlpArgs, { 
        stdio: ['ignore', 'pipe', 'pipe'] 
    });

    // 🎯 使用預建立的 FFmpeg 參數
    const ffmpegProcess = spawn(FFMPEG_EXEC_PATH, FFMPEG_ARGS, {
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // 3. 核心：將 yt-dlp 的 stdout 管道連接到 FFmpeg 的 stdin
    ytdlpProcess.stdout.pipe(ffmpegProcess.stdin);

    console.log('✅ yt-dlp 輸出已成功導向 FFmpeg 進行處理 (Piping)。');
    console.log(`--- FFmpeg 輸出管道 -> Node.js -> Redis 頻道: ${AUDIO_CHANNEL} ---`);
    
    // 4. 處理 FFmpeg 的輸出 (音頻數據) - 🎯 優化版本
    let audioBuffer = Buffer.alloc(0);
    
    ffmpegProcess.stdout.on('data', (audioChunk) => {
        // 🎯 使用更高效的 Buffer 操作
        audioBuffer = Buffer.concat([audioBuffer, audioChunk]);

        // 🎯 使用 while 迴圈處理多個完整區塊
        while (audioBuffer.length >= TARGET_CHUNK_SIZE_BYTES) {
            // 🎯 使用 subarray 比 slice 更快 (不複製，返回視圖)
            const chunkToSend = audioBuffer.subarray(0, TARGET_CHUNK_SIZE_BYTES);
            audioBuffer = audioBuffer.subarray(TARGET_CHUNK_SIZE_BYTES);

            // Base64 編碼並發佈到 Redis
            publisher.publish(AUDIO_CHANNEL, chunkToSend.toString('base64'));
        }
    });

    // 5. 🎯 改進錯誤處理：輸出 yt-dlp 的詳細錯誤
    ytdlpProcess.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg.includes('ERROR') || msg.includes('error')) {
            console.error(`[yt-dlp 錯誤]: ${msg}`);
        }
    });
    ytdlpProcess.on('error', (err) => console.error('致命錯誤：yt-dlp 啟動失敗:', err));
    ytdlpProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`yt-dlp 进程退出, Code: ${code}. 10 秒後嘗試重連...`);
            setTimeout(() => startStreamProcessing(publisher), 10000);
        }
    });
    
    // 輸出 FFmpeg 的錯誤和警告 (通常是進度信息，可以註釋掉以減少日誌)
    ffmpegProcess.stderr.on('data', (data) => {
         // console.error(`[FFmpeg 警告/錯誤]: ${data.toString().trim()}`);
    });
    ffmpegProcess.on('error', (err) => console.error('致命錯誤：FFmpeg 啟動失敗:', err));
    ffmpegProcess.on('close', (code) => {
        if (code !== 0) {
            console.log(`FFmpeg 进程退出, Code: ${code}.`);
        }
    });
}

// 3. 獲取直播 URL (yt-dlp 邏輯)
function getStreamUrl(callback) {
    console.log(`--- 正在使用 yt-dlp 解析直播串流 URL: ${LIVE_PAGE_URL} ---`);
    const YTDLP_EXEC_PATH = 'yt-dlp'; 
    const ytdlp = spawn(YTDLP_EXEC_PATH, ['-f', 'bestaudio', '--get-url', LIVE_PAGE_URL]);
    
    let streamUrl = '';
    let ytdlpError = ''; // 🌟 新增：捕獲 yt-dlp 錯誤輸出
    
    ytdlp.stdout.on('data', (data) => {
        streamUrl += data.toString().trim();
    });
    
    ytdlp.stderr.on('data', (data) => {
        // 🌟 捕獲所有 stderr 數據
        ytdlpError += data.toString();
        // console.error(`[yt-dlp 调试/警告]: ${data.toString().trim()}`); // 可以取消註釋這行查看進度
    });

    ytdlp.on('close', (code) => {
        if (code === 0 && streamUrl) {
            console.log('--- yt-dlp 解析成功。');
            callback(streamUrl);
        } else {
            // 🌟 如果退出碼不是 0 或沒有返回 URL，輸出詳細錯誤
            console.error(`致命錯誤：yt-dlp 进程退出, Code: ${code}.`);
            if (ytdlpError.trim()) {
                console.error(`yt-dlp 錯誤輸出 (stderr):\n${ytdlpError.trim()}`);
            } else {
                console.error('yt-dlp 沒有返回詳細錯誤訊息。可能原因：連結無效或非直播，或 Docker 網路問題。');
            }
            // 10 秒後重試
            setTimeout(() => getStreamUrl(callback), 10000); 
        }
    });

    ytdlp.on('error', (err) => {
        console.error('致命錯誤：yt-dlp 啟動失敗:', err);
        setTimeout(() => getStreamUrl(callback), 10000); 
    });
}

function startMainFlow() {
    initializeRedisClients();
    // 直接啟動管道處理，不再需要獲取臨時 URL
    startStreamProcessing(publisher); 
}