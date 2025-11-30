const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8081');

ws.on('open', function open() {
    console.log('Connected to WebSocket server');
    // Send a test message
    ws.send(JSON.stringify({
        type: 'chat',
        message: 'Hello from test client'
    }));
});

ws.on('message', function message(data) {
    console.log('Received:', data.toString());
});

ws.on('error', function error(err) {
    console.error('WebSocket error:', err);
});

ws.on('close', function close(code, reason) {
    console.log('Connection closed:', code, reason.toString());
});

// Timeout after 10 seconds
setTimeout(() => {
    ws.close();
    console.log('Test completed');
}, 10000);