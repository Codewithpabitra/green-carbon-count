// Method 1: Using child_process to run Python FastAPI as subprocess
const express = require('express');
const { spawn } = require('child_process');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const router = express.Router();

// Configuration
const PYTHON_API_PORT = 8001; // Different port to avoid conflicts
const PYTHON_API_URL = `http://localhost:${PYTHON_API_PORT}`;

// Global variable to track Python process
let pythonProcess = null;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Start Python FastAPI server
function startPythonAPI() {
    return new Promise((resolve, reject) => {
        console.log('🚀 Starting Python ML API...');
        
        // Spawn Python process
        pythonProcess = spawn('python', ['main.py'], {
            env: { ...process.env, PORT: PYTHON_API_PORT },
            stdio: ['pipe', 'pipe', 'pipe']
        });

        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python API: ${data.toString()}`);
            if (data.toString().includes('Uvicorn running')) {
                resolve();
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python API Error: ${data.toString()}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python API process exited with code ${code}`);
            pythonProcess = null;
        });

        // Timeout fallback
        setTimeout(() => {
            resolve(); // Assume it started after timeout
        }, 5000);
    });
}

// Proxy middleware to forward requests to Python API
async function proxyToPython(req, res, next) {
    try {
        const pythonUrl = `${PYTHON_API_URL}${req.originalUrl.replace('/ml', '')}`;
        
        let response;
        if (req.method === 'GET') {
            response = await axios.get(pythonUrl, {
                params: req.query,
                timeout: 30000
            });
        } else if (req.method === 'POST') {
            response = await axios.post(pythonUrl, req.body, {
                headers: { 'Content-Type': 'application/json' },
                timeout: 30000
            });
        }

        res.json(response.data);
    } catch (error) {
        console.error('Python API proxy error:', error.message);
        res.status(500).json({ 
            error: 'ML service unavailable', 
            detail: error.message 
        });
    }
}

// ML Routes - All under /ml prefix
router.get('/health', proxyToPython);
router.post('/upload', proxyToPython);
router.get('/data/monthly', proxyToPython);
router.get('/data/daily', proxyToPython);
router.get('/stats', proxyToPython);
router.get('/stats/comparison', proxyToPython);
router.get('/stats/peak-hours', proxyToPython);
router.get('/leakage/alerts', proxyToPython);
router.post('/predict', proxyToPython);
router.post('/predict-next-month', proxyToPython);
router.post('/predict-month-after-next', proxyToPython);
router.post('/retrain', proxyToPython);
router.get('/retrain/status', proxyToPython);
router.get('/download/:filename', proxyToPython);
router.get('/list-files', proxyToPython);

// Mount ML routes
app.use('/ml/api', router);

// Serve the frontend at /ml
app.get('/ml', (req, res) => {
    res.sendFile(path.join(__dirname, ""));
});

// Your existing routes
app.get('/api/users', (req, res) => {
    // Your existing user routes
    res.json({ message: 'Users endpoint' });
});

// Start server
async function startServer() {
    try {
        // Start Python API first
        await startPythonAPI();
        console.log('✅ Python ML API started');
        
        // Start Express server
        const PORT = process.env.PORT || 3000;
        app.listen(PORT, () => {
            console.log(`🌐 Express server running on port ${PORT}`);
            console.log(`📊 ML Analytics available at: http://localhost:${PORT}/ml`);
            console.log(`🔌 ML API proxy at: http://localhost:${PORT}/ml/api/*`);
        });
    } catch (error) {
        console.error('❌ Failed to start services:', error);
        process.exit(1);
    }
}

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down gracefully...');
    
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM');
    }
    
    process.exit(0);
});

// Start the application
startServer();