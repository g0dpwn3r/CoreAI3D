const path = require('path');
const axios = require('axios');
const WebSocket = require('ws');

let mainWindow;
let client = null;

// Import Electron modules
let electron;
try {
  electron = require('electron');
} catch (error) {
  console.error('Failed to load Electron:', error);
  process.exit(1);
}

// Handle the case where electron returns a path string
let app, BrowserWindow, ipcMain, dialog;
if (typeof electron === 'string') {
  // Electron returns a path string, require it as a module
  const electronModule = require(electron);
  ({ app, BrowserWindow, ipcMain, dialog } = electronModule);
} else {
  // Normal object destructuring
  ({ app, BrowserWindow, ipcMain, dialog } = electron);
}

// Check if we have valid Electron objects
if (!app || !BrowserWindow || !ipcMain) {
  console.error('Electron modules not properly loaded');
  process.exit(1);
}

// Suppress OpenGL warnings
app.commandLine.appendSwitch('disable-gpu');
app.commandLine.appendSwitch('disable-software-rasterizer');
app.commandLine.appendSwitch('disable-gpu-compositing');
app.commandLine.appendSwitch('disable-gpu-rasterization');
app.commandLine.appendSwitch('disable-gpu-sandbox');
app.commandLine.appendSwitch('no-sandbox');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: path.join(__dirname, 'assets', 'icon.ico'),
    title: 'CoreAI3D Dashboard'
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Handle app ready event
app.whenReady().then(createWindow).catch(error => {
  console.error('Failed to initialize app:', error);
  process.exit(1);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers for client operations
ipcMain.handle('initialize-client', async (event, config) => {
  try {
    client = {
      apiUrl: config.apiUrl,
      wsUrl: config.wsUrl,
      apiKey: config.apiKey,
      axios: axios.create({
        baseURL: config.apiUrl,
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      })
    };
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('test-connection', async () => {
  try {
    if (!client) {
      return { success: false, error: 'Client not initialized' };
    }

    const response = await client.axios.get('/health');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-system-metrics', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/system/metrics');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('start-training', async (event, config) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/training/start', config);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-training', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/training/stop');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('pause-training', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/training/pause');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('resume-training', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/training/resume');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('export-training-logs', async (event, logs) => {
  try {
    const fs = require('fs');
    const path = require('path');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `training_logs_${timestamp}.txt`;
    const filePath = path.join(require('os').homedir(), 'Desktop', filename);

    fs.writeFileSync(filePath, logs);
    return { success: true, path: filePath };
  } catch (error) {
    return { success: false, error: error.message };
  }
});


ipcMain.handle('load-model', async (event, identifier, type) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/models/load', { identifier, type });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('generate-code', async (event, data) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/code/generate', data);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('chat-message', async (event, message) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/chat', { message });
    return { success: true, response: response.data.response };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-neural-topology', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/neural/topology');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-neural-activity', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/neural/activity');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// File dialog handlers
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile']
  });
  return result.filePaths[0] || null;
});

ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  return result.filePaths[0] || null;
});

ipcMain.handle('save-file', async (event, content, filename) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: filename
  });

  if (!result.canceled) {
    const fs = require('fs');
    fs.writeFileSync(result.filePath, content);
    return { success: true, path: result.filePath };
  }

  return { success: false };
});

// Training data management
ipcMain.handle('create-dataset', async (event, name, description, dataType) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/datasets/create', { name, description, dataType });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('delete-dataset', async (event, name) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.delete(`/datasets/${name}`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('add-file-to-dataset', async (event, datasetName, filePath) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post(`/datasets/${datasetName}/files`, { filePath });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Testing functionality
ipcMain.handle('run-test', async (event, testType, testModule) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/tests/run', { testType, testModule });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('run-load-test', async (event, concurrentUsers, duration, endpoint) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/performance/load_test', {
      concurrentUsers,
      duration,
      endpoint
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Prediction functionality
ipcMain.handle('run-prediction', async (event, modelType, specificModel, inputType, inputData, containsHeader, containsText) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/predict', {
      modelType, specificModel, inputType, inputData, containsHeader, containsText
    });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Sandbox functionality
ipcMain.handle('start-sandbox', async (event, config) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/sandbox/start', config);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-sandbox', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/sandbox/stop');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Debugging functionality
ipcMain.handle('get-system-info', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/system/info');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-process-list', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/system/processes');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('run-api-debug', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/debug/api');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('run-system-debug', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/debug/system');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Linux sandbox functionality
ipcMain.handle('execute-linux-command', async (event, { command, container }) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/linux/execute', { command, container });
    return { success: true, output: response.data.output };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('start-linux-container', async (event, config) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/linux/container/start', config);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-linux-container', async (event, containerId) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/linux/container/stop', { containerId });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('run-linux-training', async (event, scenario) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/linux/training/run', { scenario });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('generate-training-report', async (event, trainingId) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/linux/training/report', { trainingId });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('refresh-containers', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/linux/containers');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Downloads functionality
ipcMain.handle('search-models', async (event, { searchTerm, typeFilter, page = 1, source = 'all' }) => {
  try {
    const models = [];

    // Search Kaggle models
    if (source === 'all' || source === 'kaggle') {
      const kaggleModels = await searchKaggleModels(searchTerm, typeFilter, page);
      models.push(...kaggleModels);
    }

    // Search HuggingFace models
    if (source === 'all' || source === 'huggingface') {
      const hfModels = await searchHuggingFaceModels(searchTerm, typeFilter, page);
      models.push(...hfModels);
    }

    return { success: true, models, hasMore: models.length >= 20 };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('refresh-models', async (event, { page = 1, source = 'all' }) => {
  try {
    const models = [];

    // Get latest Kaggle models
    if (source === 'all' || source === 'kaggle') {
      const kaggleModels = await getKaggleModels(page);
      models.push(...kaggleModels);
    }

    // Get latest HuggingFace models
    if (source === 'all' || source === 'huggingface') {
      const hfModels = await getHuggingFaceModels(page);
      models.push(...hfModels);
    }

    return { success: true, models, hasMore: models.length >= 20 };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Kaggle API integration
async function searchKaggleModels(searchTerm, typeFilter, page = 1) {
  try {
    const searchQuery = searchTerm ? `search=${encodeURIComponent(searchTerm)}` : '';
    const typeQuery = typeFilter && typeFilter !== 'all' ? `&type=${typeFilter}` : '';
    const pageQuery = `&page=${page}`;

    const url = `https://www.kaggle.com/api/v1/models/list?${searchQuery}${typeQuery}${pageQuery}&sortBy=hotness&group=public`;

    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'CoreAI3D-Dashboard/1.0'
      },
      timeout: 10000
    });

    return response.data.map(model => ({
      id: `kaggle-${model.id}`,
      name: model.title || model.ref,
      type: getModelType(model.tags || []),
      source: 'Kaggle',
      size: model.totalSize ? `${(model.totalSize / (1024 * 1024)).toFixed(1)}MB` : 'N/A',
      downloads: model.totalDownloads || 0,
      description: model.description || '',
      url: `https://www.kaggle.com/${model.ref}`,
      lastUpdated: model.lastUpdated || model.updated
    }));
  } catch (error) {
    console.error('Kaggle API search error:', error.message);
    return [];
  }
}

async function getKaggleModels(page = 1) {
  try {
    const url = `https://www.kaggle.com/api/v1/models/list?page=${page}&sortBy=hotness&group=public`;

    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'CoreAI3D-Dashboard/1.0'
      },
      timeout: 10000
    });

    return response.data.map(model => ({
      id: `kaggle-${model.id}`,
      name: model.title || model.ref,
      type: getModelType(model.tags || []),
      source: 'Kaggle',
      size: model.totalSize ? `${(model.totalSize / (1024 * 1024)).toFixed(1)}MB` : 'N/A',
      downloads: model.totalDownloads || 0,
      description: model.description || '',
      url: `https://www.kaggle.com/${model.ref}`,
      lastUpdated: model.lastUpdated || model.updated
    }));
  } catch (error) {
    console.error('Kaggle API error:', error.message);
    return [];
  }
}

// HuggingFace API integration
async function searchHuggingFaceModels(searchTerm, typeFilter, page = 1) {
  try {
    const searchQuery = searchTerm ? `search=${encodeURIComponent(searchTerm)}` : '';
    const typeQuery = typeFilter && typeFilter !== 'all' ? `&pipeline_tag=${typeFilter}` : '';
    const pageQuery = `&limit=20&offset=${(page - 1) * 20}`;

    const url = `https://huggingface.co/api/models?${searchQuery}${typeQuery}${pageQuery}&sort=downloads&direction=-1`;

    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'CoreAI3D-Dashboard/1.0'
      },
      timeout: 10000
    });

    return response.data.map(model => ({
      id: `hf-${model.id.replace('/', '-')}`,
      name: model.id,
      type: getHFModelType(model.pipeline_tag || model.tags || []),
      source: 'HuggingFace',
      size: 'N/A', // HF doesn't provide size in basic API
      downloads: model.downloads || 0,
      description: model.description || '',
      url: `https://huggingface.co/${model.id}`,
      lastModified: model.lastModified
    }));
  } catch (error) {
    console.error('HuggingFace API search error:', error.message);
    return [];
  }
}

async function getHuggingFaceModels(page = 1) {
  try {
    const url = `https://huggingface.co/api/models?limit=20&offset=${(page - 1) * 20}&sort=downloads&direction=-1`;

    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'CoreAI3D-Dashboard/1.0'
      },
      timeout: 10000
    });

    return response.data.map(model => ({
      id: `hf-${model.id.replace('/', '-')}`,
      name: model.id,
      type: getHFModelType(model.pipeline_tag || model.tags || []),
      source: 'HuggingFace',
      size: 'N/A',
      downloads: model.downloads || 0,
      description: model.description || '',
      url: `https://huggingface.co/${model.id}`,
      lastModified: model.lastModified
    }));
  } catch (error) {
    console.error('HuggingFace API error:', error.message);
    return [];
  }
}

// Helper functions
function getModelType(tags) {
  if (!Array.isArray(tags)) return 'Other';

  const tagString = tags.join(' ').toLowerCase();

  if (tagString.includes('vision') || tagString.includes('image') || tagString.includes('computer-vision')) return 'Vision';
  if (tagString.includes('text') || tagString.includes('nlp') || tagString.includes('language')) return 'Text';
  if (tagString.includes('audio') || tagString.includes('speech')) return 'Audio';
  if (tagString.includes('multimodal') || tagString.includes('multi-modal')) return 'Multimodal';

  return 'Other';
}

function getHFModelType(pipelineTag) {
  if (!pipelineTag) return 'Other';

  const tag = Array.isArray(pipelineTag) ? pipelineTag[0] : pipelineTag;

  switch (tag) {
    case 'text-classification':
    case 'token-classification':
    case 'question-answering':
    case 'text-generation':
    case 'text2text-generation':
    case 'summarization':
    case 'translation':
    case 'conversational':
      return 'Text';
    case 'image-classification':
    case 'object-detection':
    case 'image-segmentation':
    case 'image-to-image':
      return 'Vision';
    case 'text-to-speech':
    case 'automatic-speech-recognition':
    case 'audio-classification':
      return 'Audio';
    case 'image-text-to-text':
    case 'visual-question-answering':
      return 'Multimodal';
    default:
      return 'Other';
  }
}

ipcMain.handle('download-model', async (event, modelId) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/models/download', { modelId });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('cancel-model-download', async (event, downloadId) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/models/download/cancel', { downloadId });
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('download-fasttext', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/models/download/fasttext');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Code generation functionality
ipcMain.handle('generate-code', async (event, data) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/code/generate', data);
    return { success: true, code: response.data.code };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Chat functionality
ipcMain.handle('chat-message', async (event, message) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.post('/chat', { message });
    return { success: true, response: response.data.response };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Neural network functionality
ipcMain.handle('get-neural-topology', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/neural/topology');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-neural-activity', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/neural/activity');
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});