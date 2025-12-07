const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
// 引入 Redis
const Redis = require('ioredis'); 

// 日誌開關：預設只顯示錯誤；設 LOG_VERBOSE=1 以查看資訊級別
const LOG_VERBOSE = process.env.LOG_VERBOSE === '1';
const log = (...args) => LOG_VERBOSE && console.log(...args);

// --- 配置參數 (預先計算的常數) ---
const WSS_PORT = 8080; 
const LIVE_PAGE_URL = 'https://www.twitch.tv/tenshiuyu'; // 直播頁面 URL

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

// 🎯 預先建立的 Redis 連線選項 (優化連線速度)
const REDIS_OPTIONS = Object.freeze({
    host: REDIS_HOST,
    port: REDIS_PORT,
    retryStrategy: (times) => Math.min(times * 50, 2000),
    maxRetriesPerRequest: 3,
    enableReadyCheck: false,
    lazyConnect: true,       // 🚀 延遲連線，加快啟動
    connectTimeout: 5000,    // 🚀 縮短連線超時
    commandTimeout: 3000,    // 🚀 命令超時
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

// 🎯 同步預讀 client.html (伺服器啟動時即就緒)
const CLIENT_HTML_PATH = path.join(__dirname, '../client.html');
let cachedClientHtml;
try {
    cachedClientHtml = fs.readFileSync(CLIENT_HTML_PATH);
} catch (e) {
    console.error('⚠️ 無法預載 client.html:', e.message);
    cachedClientHtml = null;
}

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
    // 🎯 極簡路由：只處理根路徑
    if (req.url === '/' && cachedClientHtml) {
        res.writeHead(200, HTML_HEADERS);
        res.end(cachedClientHtml);
    } else if (req.url === '/') {
        res.writeHead(503);
        res.end('Service loading...');
    } else {
        res.writeHead(404);
        res.end();
    }
});

wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    log('Client connected.');
    ws.on('close', () => log('Client disconnected.'));
});

server.listen(WSS_PORT, () => {
    log(`Node.js WebSocket Server 启动在 ws://localhost:${WSS_PORT}`);
    startMainFlow();
});


// 1. 初始化 Redis 客戶端並訂閱翻譯結果
async function initializeRedisClients() {
    // 🎯 使用預建立的 Redis 選項
    publisher = new Redis(REDIS_OPTIONS);
    subscriber = new Redis(REDIS_OPTIONS);

    // 🚀 精簡事件處理器
    publisher.on('error', (err) => console.error('Redis Publisher 錯誤:', err.message));
    subscriber.on('error', (err) => console.error('Redis Subscriber 錯誤:', err.message));
    
    // 🚀 並行連線 Redis
    await Promise.all([publisher.connect(), subscriber.connect()]);
    log('✅ Redis 連線就緒');

    // 訂閱翻譯結果頻道 (來自 Python)
    subscriber.subscribe(TRANSLATION_CHANNEL, (err, count) => {
        if (err) {
            console.error('致命錯誤：Redis 訂閱翻譯頻道失敗:', err);
            return;
        }
        log(`Node.js 成功訂閱 Redis 頻道: ${TRANSLATION_CHANNEL} (${count} 個頻道)。`);
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
    log(`--- 正在使用 yt-dlp 啟動串流處理: ${LIVE_PAGE_URL} ---`);
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

    log('✅ yt-dlp 輸出已成功導向 FFmpeg 進行處理 (Piping)。');
    log(`--- FFmpeg 輸出管道 -> Node.js -> Redis 頻道: ${AUDIO_CHANNEL} ---`);
    
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
            console.error(`FFmpeg 进程退出, Code: ${code}.`);
        }
    });
}

async function startMainFlow() {
    await initializeRedisClients();
    // 🚀 Redis 就緒後立即啟動串流處理
    startStreamProcessing(publisher); 
}