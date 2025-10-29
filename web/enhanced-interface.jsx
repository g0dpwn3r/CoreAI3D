import React, { useState, useEffect, useRef } from 'react';

// Enhanced AI Interface with API Integration
const EnhancedAIInterface = () => {
    // API Configuration
    const [apiConfig, setApiConfig] = useState({
        baseURL: 'http://0.0.0.0:8080/api/v1',
        wsURL: 'ws://0.0.0.0:8081/ws',
        sessionId: '',
        apiKey: '',
        timeout: 30000
    });

    // Chat state
    const [messages, setMessages] = useState([]);
    const [currentMessage, setCurrentMessage] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isTyping, setIsTyping] = useState(false);

    // File upload state
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isUploading, setIsUploading] = useState(false);

    // Real-time processing state
    const [activeStreams, setActiveStreams] = useState({
        vision: false,
        audio: false,
        system: false,
        web: false
    });

    // Module status
    const [moduleStatus, setModuleStatus] = useState({
        vision: { available: false, active: false },
        audio: { available: false, active: false },
        system: { available: false, active: false },
        web: { available: false, active: false },
        math: { available: false, active: false }
    });

    // WebSocket connection
    const wsRef = useRef(null);
    const messagesEndRef = useRef(null);

    // API utility functions
    const apiCall = async (endpoint, method = 'GET', data = null) => {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiConfig.apiKey}`,
                'X-Session-ID': apiConfig.sessionId
            }
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            config.body = JSON.stringify(data);
        }

        const response = await fetch(`${apiConfig.baseURL}${endpoint}`, config);
        return await response.json();
    };

    const uploadFile = async (file) => {
        setIsUploading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('fileType', file.type);
            formData.append('description', `Upload of ${file.name}`);

            const response = await fetch(`${apiConfig.baseURL}/files/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiConfig.apiKey}`,
                    'X-Session-ID': apiConfig.sessionId
                },
                body: formData
            });

            const result = await response.json();
            setUploadedFiles(prev => [...prev, { ...result, file }]);
            return result;
        } catch (error) {
            console.error('File upload error:', error);
            throw error;
        } finally {
            setIsUploading(false);
        }
    };

    // WebSocket functions
    const connectWebSocket = () => {
        if (wsRef.current) {
            wsRef.current.close();
        }

        wsRef.current = new WebSocket(apiConfig.wsURL);

        wsRef.current.onopen = () => {
            setIsConnected(true);
            console.log('WebSocket connected');
        };

        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };

        wsRef.current.onclose = () => {
            setIsConnected(false);
            console.log('WebSocket disconnected');
        };

        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    };

    const handleWebSocketMessage = (data) => {
        switch (data.type) {
            case 'chat_response':
                addMessage(data.content, 'ai', data.metadata);
                break;
            case 'stream_data':
                handleStreamData(data);
                break;
            case 'system_status':
                updateModuleStatus(data.modules);
                break;
            case 'error':
                addMessage(`Error: ${data.message}`, 'error');
                break;
            default:
                console.log('Unknown message type:', data);
        }
    };

    const sendWebSocketMessage = (message) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        }
    };

    // Message handling
    const addMessage = (content, sender = 'user', metadata = {}) => {
        const message = {
            id: Date.now().toString(),
            content,
            sender,
            timestamp: new Date().toISOString(),
            metadata
        };
        setMessages(prev => [...prev, message]);
    };

    const sendMessage = async () => {
        if (!currentMessage.trim()) return;

        const userMessage = currentMessage;
        setCurrentMessage('');
        addMessage(userMessage, 'user');

        setIsTyping(true);

        try {
            // Send via WebSocket for real-time response
            sendWebSocketMessage({
                type: 'chat_message',
                content: userMessage,
                timestamp: new Date().toISOString()
            });

            // Also send via REST API for backup
            const response = await apiCall('/chat/message', 'POST', {
                message: userMessage,
                messageType: 'text'
            });

            if (response.success) {
                // Response will come via WebSocket
            } else {
                addMessage(`API Error: ${response.message}`, 'error');
            }
        } catch (error) {
            console.error('Send message error:', error);
            addMessage('Failed to send message. Please try again.', 'error');
        } finally {
            setIsTyping(false);
        }
    };

    const handleStreamData = (data) => {
        // Handle real-time stream data (vision, audio, etc.)
        switch (data.streamType) {
            case 'vision':
                // Update vision stream display
                break;
            case 'audio':
                // Update audio visualization
                break;
            case 'system':
                // Update system monitoring
                break;
            default:
                console.log('Stream data:', data);
        }
    };

    const updateModuleStatus = (modules) => {
        setModuleStatus(prev => ({ ...prev, ...modules }));
    };

    // Real-time processing functions
    const startVisionStream = async (source = 'camera') => {
        try {
            const response = await apiCall('/streams/vision/start', 'POST', {
                source,
                frameRate: 30
            });

            if (response.success) {
                setActiveStreams(prev => ({ ...prev, vision: true }));
                sendWebSocketMessage({
                    type: 'start_stream',
                    streamType: 'vision',
                    source
                });
            }
        } catch (error) {
            console.error('Start vision stream error:', error);
        }
    };

    const startAudioStream = async (source = 'microphone') => {
        try {
            const response = await apiCall('/streams/audio/start', 'POST', {
                source,
                sampleRate: 44100
            });

            if (response.success) {
                setActiveStreams(prev => ({ ...prev, audio: true }));
                sendWebSocketMessage({
                    type: 'start_stream',
                    streamType: 'audio',
                    source
                });
            }
        } catch (error) {
            console.error('Start audio stream error:', error);
        }
    };

    const startSystemMonitoring = async (interval = 1000) => {
        try {
            const response = await apiCall('/monitoring/system/start', 'POST', {
                interval
            });

            if (response.success) {
                setActiveStreams(prev => ({ ...prev, system: true }));
                sendWebSocketMessage({
                    type: 'start_monitoring',
                    monitoringType: 'system',
                    interval
                });
            }
        } catch (error) {
            console.error('Start system monitoring error:', error);
        }
    };

    // File handling
    const handleFileUpload = async (event) => {
        const files = Array.from(event.target.files);
        for (const file of files) {
            try {
                await uploadFile(file);
            } catch (error) {
                console.error('File upload error:', error);
            }
        }
    };

    const processFileWithAI = async (fileId, analysisTypes) => {
        try {
            const response = await apiCall('/multimodal/analyze', 'POST', {
                fileId,
                analysisTypes
            });

            if (response.success) {
                addMessage(`File analyzed: ${response.result.summary}`, 'ai', {
                    analysis: response.result
                });
            }
        } catch (error) {
            console.error('File analysis error:', error);
        }
    };

    // Command execution
    const executeCommand = async (command, parameters = {}) => {
        try {
            const response = await apiCall('/commands/execute', 'POST', {
                command,
                parameters
            });

            if (response.success) {
                addMessage(`Command executed: ${command}`, 'ai', {
                    result: response.result
                });
            } else {
                addMessage(`Command failed: ${response.message}`, 'error');
            }
        } catch (error) {
            console.error('Command execution error:', error);
        }
    };

    // Initialize connection
    useEffect(() => {
        connectWebSocket();
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    // Auto-scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <div className="w-80 bg-gray-800 p-4 flex flex-col">
                <h2 className="text-xl font-bold mb-4 text-blue-400">AI Control Panel</h2>

                {/* Connection Status */}
                <div className="mb-4">
                    <div className="flex items-center mb-2">
                        <div className={`w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                        <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
                    </div>
                    <button
                        onClick={connectWebSocket}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded text-sm"
                    >
                        Reconnect
                    </button>
                </div>

                {/* Module Status */}
                <div className="mb-4">
                    <h3 className="text-lg font-semibold mb-2">Modules</h3>
                    {Object.entries(moduleStatus).map(([module, status]) => (
                        <div key={module} className="flex items-center justify-between mb-1">
                            <span className="text-sm capitalize">{module}</span>
                            <div className="flex space-x-1">
                                <div className={`w-2 h-2 rounded-full ${status.available ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                                <div className={`w-2 h-2 rounded-full ${status.active ? 'bg-blue-500' : 'bg-gray-500'}`}></div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Active Streams */}
                <div className="mb-4">
                    <h3 className="text-lg font-semibold mb-2">Active Streams</h3>
                    {Object.entries(activeStreams).map(([stream, active]) => (
                        <div key={stream} className="flex items-center justify-between mb-1">
                            <span className="text-sm capitalize">{stream}</span>
                            <button
                                onClick={() => active ? stopStream(stream) : startStream(stream)}
                                className={`px-2 py-1 rounded text-xs ${
                                    active ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                                }`}
                            >
                                {active ? 'Stop' : 'Start'}
                            </button>
                        </div>
                    ))}
                </div>

                {/* File Upload */}
                <div className="mb-4">
                    <h3 className="text-lg font-semibold mb-2">Files</h3>
                    <input
                        type="file"
                        multiple
                        onChange={handleFileUpload}
                        disabled={isUploading}
                        className="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                    />
                    {isUploading && <div className="text-sm text-blue-400 mt-1">Uploading...</div>}
                </div>

                {/* Quick Actions */}
                <div className="flex-1">
                    <h3 className="text-lg font-semibold mb-2">Quick Actions</h3>
                    <div className="space-y-2">
                        <button
                            onClick={() => executeCommand('system.get_metrics')}
                            className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded text-sm"
                        >
                            Get System Metrics
                        </button>
                        <button
                            onClick={() => executeCommand('vision.analyze_screen')}
                            className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded text-sm"
                        >
                            Analyze Screen
                        </button>
                        <button
                            onClick={() => executeCommand('web.search', { query: 'latest news' })}
                            className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded text-sm"
                        >
                            Search Web
                        </button>
                    </div>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-gray-800 p-4 border-b border-gray-700">
                    <h1 className="text-2xl font-bold text-blue-400">AI Assistant</h1>
                    <p className="text-gray-400">Multi-modal AI with real-time processing</p>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((message) => (
                        <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                                message.sender === 'user'
                                    ? 'bg-blue-600 text-white'
                                    : message.sender === 'error'
                                    ? 'bg-red-600 text-white'
                                    : 'bg-gray-700 text-gray-100'
                            }`}>
                                <div className="text-sm">{message.content}</div>
                                <div className="text-xs text-gray-400 mt-1">
                                    {new Date(message.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                        </div>
                    ))}
                    {isTyping && (
                        <div className="flex justify-start">
                            <div className="bg-gray-700 text-gray-100 px-4 py-2 rounded-lg">
                                <div className="flex space-x-1">
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Message Input */}
                <div className="bg-gray-800 p-4 border-t border-gray-700">
                    <div className="flex space-x-2">
                        <input
                            type="text"
                            value={currentMessage}
                            onChange={(e) => setCurrentMessage(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                            placeholder="Type your message..."
                            className="flex-1 bg-gray-700 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            disabled={!isConnected}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={!isConnected || !currentMessage.trim()}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg"
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EnhancedAIInterface;