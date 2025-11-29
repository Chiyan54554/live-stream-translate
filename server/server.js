const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
// 引入 Redis
const Redis = require('ioredis'); 

// --- 配置參數 ---
const WSS_PORT = 8080;
const LIVE_PAGE_URL = 'https://www.twitch.tv/amamiyakokoro_'; // 直播頁面 URL

// Redis 配置
const REDIS_HOST = process.env.REDIS_HOST || 'localhost'; 
const REDIS_PORT = parseInt(process.env.REDIS_PORT) || 6379; 

const AUDIO_CHANNEL = "audio_feed";           // Node.js -> Python (發佈音頻)
const TRANSLATION_CHANNEL = "translation_feed"; // Python -> Node.js (訂閱翻譯)

const SAMPLE_RATE = 16000;
const BYTES_PER_SAMPLE = 2; // 16-bit PCM = 2 Bytes

// 定義每個音訊塊的時長 (決定 Redis 發佈頻率)
const CHUNK_DURATION_S = 0.128; // 128 毫秒

// 計算 Node.js 每次發佈到 Redis 所需的位元組數
const TARGET_CHUNK_SIZE_BYTES = Math.ceil(
    CHUNK_DURATION_S * SAMPLE_RATE * BYTES_PER_SAMPLE
);

let ffmpegProcess = null;
let publisher; // Redis publisher client
let subscriber; // Redis subscriber client
let wss; 

// [ WebSocket 啟動和連線邏輯 ]
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
                // 數據是乾淨的 JSON 字符串，直接廣播
                JSON.parse(message); 
                wss.clients.forEach(client => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(message); 
                    }
                });
            } catch (error) {
                console.error('致命錯誤：無法解析 Redis 接收到的 JSON 數據:', error.message);
            }
        }
    });
}

// 2. 啟動串流處理 (yt-dlp -> Pipe -> FFmpeg -> Redis)
function startStreamProcessing(publisher) {
    console.log(`--- 正在使用 yt-dlp 啟動串流處理: ${LIVE_PAGE_URL} ---`);
    const YTDLP_EXEC_PATH = 'yt-dlp';
    const FFMPEG_EXEC_PATH = 'ffmpeg';
    
    // 1. 啟動 yt-dlp，要求它將音頻流輸出到 stdout ('-o', '-')
    // 我們同時要求 yt-dlp 輸出日誌到 stderr，方便除錯。
    const ytdlpProcess = spawn(YTDLP_EXEC_PATH, [
        '-f', 'bestaudio', 
        '--no-warnings',
        '--force-ipv4',
        '--referer', 'https://www.twitch.tv/',  // 模擬從網頁發起連線
        '--no-check-certificate',               // 忽略 SSL/TLS 證書檢查 (有時能解決握手問題)
        '-o', '-', 
        LIVE_PAGE_URL
    ], { stdio: ['ignore', 'pipe', 'pipe'] });

    // 2. 啟動 FFmpeg，從 stdin 讀取音頻 ('-i', 'pipe:0')
    const ffmpegArgs = [
        '-i', 'pipe:0',          // 讓 FFmpeg 從其 stdin 讀取數據 (即 yt-dlp 的輸出)
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-f', 's16le',
        'pipe:1'                 // 輸出到 stdout
    ];

    const ffmpegProcess = spawn(FFMPEG_EXEC_PATH, ffmpegArgs, {
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // 3. 核心：將 yt-dlp 的 stdout 管道連接到 FFmpeg 的 stdin
    ytdlpProcess.stdout.pipe(ffmpegProcess.stdin);

    console.log('✅ yt-dlp 輸出已成功導向 FFmpeg 進行處理 (Piping)。');
    console.log(`--- FFmpeg 輸出管道 -> Node.js -> Redis 頻道: ${AUDIO_CHANNEL} ---`);
    
    // 4. 處理 FFmpeg 的輸出 (音頻數據) - 【關鍵修改區】
    let audioBuffer = Buffer.alloc(0); // 緩衝器：用於累積數據
    
    ffmpegProcess.stdout.on('data', (audioChunk) => {
        // 1. 將新收到的音訊數據追加到緩衝區
        audioBuffer = Buffer.concat([audioBuffer, audioChunk]);

        // 2. 循環檢查緩衝區是否達到目標塊大小
        while (audioBuffer.length >= TARGET_CHUNK_SIZE_BYTES) {
            // a. 擷取固定大小的音訊塊
            const chunkToSend = audioBuffer.slice(0, TARGET_CHUNK_SIZE_BYTES);
            
            // b. 移除已發送的數據
            audioBuffer = audioBuffer.slice(TARGET_CHUNK_SIZE_BYTES);

            // c. Base64 編碼並發佈到 Redis
            const base64Audio = chunkToSend.toString('base64');
            publisher.publish(AUDIO_CHANNEL, base64Audio).catch(err => {
                console.error('致命錯誤：發佈音頻數據到 Redis 失敗:', err);
            });
        }
    });
    
    // 5. 錯誤和關閉處理
    // 輸出 yt-dlp 的錯誤和警告
    ytdlpProcess.stderr.on('data', (data) => {
        // 大部分是警告，但仍需注意
        // console.error(`[yt-dlp 警告/錯誤]: ${data.toString().trim()}`); 
    });
    ytdlpProcess.on('error', (err) => console.error('致命錯誤：yt-dlp 啟動失敗:', err));
    ytdlpProcess.on('close', (code) => {
        if (code !== 0) console.error(`yt-dlp 进程退出, Code: ${code}. 直播流可能中斷。`);
    });
    
    // 輸出 FFmpeg 的錯誤和警告 (通常是進度信息，可以註釋掉以減少日誌)
    ffmpegProcess.stderr.on('data', (data) => {
         // console.error(`[FFmpeg 警告/錯誤]: ${data.toString().trim()}`);
    });
    ffmpegProcess.on('error', (err) => console.error('致命錯誤：FFmpeg 啟動失敗:', err));
    ffmpegProcess.on('close', (code) => {
        console.log(`FFmpeg 进程退出, Code: ${code}. 正在嘗試重連...`);
        // 注意：這裡可以加入重連邏輯，但目前先以啟動成功為目標。
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