import websocket
import json
import threading

def on_message(ws, message):
    print('Received:', message)

def on_error(ws, error):
    print('WebSocket error:', error)

def on_close(ws, close_status_code, close_msg):
    print('Connection closed:', close_status_code, close_msg)

def on_open(ws):
    print('Connected to WebSocket server')
    # Send a test message
    ws.send(json.dumps({
        'type': 'chat',
        'message': 'Hello from test client'
    }))
    # Close after 10 seconds
    threading.Timer(10, ws.close).start()

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:8081",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
    print('Test completed')