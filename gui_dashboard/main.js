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

// Add fallback for whenReady if not available
if (!app.whenReady) {
  app.whenReady = () => new Promise(resolve => {
    if (app.isReady()) {
      resolve();
    } else {
      app.once('ready', resolve);
    }
  });
}

// Check if we have valid Electron objects
if (!app || !BrowserWindow || !ipcMain) {
  console.error('Electron modules not properly loaded');
  process.exit(1);
}

// Check if Electron is properly loaded
if (!app) {
  console.error('Electron app module not found. Make sure Electron is properly installed.');
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
if (app) {
  app.whenReady().then(createWindow).catch(error => {
    console.error('Failed to initialize app:', error);
    process.exit(1);
  });
} else {
  console.error('Electron app object is not available');
  process.exit(1);
}

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
ipcMain.handle('search-models', async (event, { searchTerm, typeFilter }) => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/models/search', { params: { q: searchTerm, type: typeFilter } });
    return { success: true, models: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('refresh-models', async () => {
  try {
    if (!client) return { success: false, error: 'Client not initialized' };
    const response = await client.axios.get('/models/refresh');
    return { success: true, models: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

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