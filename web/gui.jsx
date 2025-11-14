import React, { useState, useEffect } from 'react';

// Main App Component
const App = () => {
    // State for all parameters
    const [inputFile, setInputFile] = useState('');
    const [targetFile, setTargetFile] = useState('');
    const [outputCsv, setOutputCsv] = useState('');
    const [delimiter, setDelimiter] = useState(',');
    const [numSamples, setNumSamples] = useState(-1);
    const [language, setLanguage] = useState('en');
    const [embeddingFile, setEmbeddingFile] = useState('embedding.txt');
    const [epochs, setEpochs] = useState(10);
    const [learningRate, setLearningRate] = useState(0.01);
    const [layers, setLayers] = useState(3);
    const [neurons, setNeurons] = useState(10);
    const [minRange, setMinRange] = useState(0.0);
    const [maxRange, setMaxRange] = useState(1.0);
    const [dbHost, setDbHost] = useState('0.0.0.0');
    const [dbPort, setDbPort] = useState(33060);
    const [dbUser, setDbUser] = useState('user');
    const [dbPassword, setDbPassword] = useState('password');
    const [dbSchema, setDbSchema] = useState('coreai_db');
    const [sslMode, setSslMode] = useState('DISABLED'); // DISABLED, VERIFY_CA, VERIFY_IDENTITY
    const [createTables, setCreateTables] = useState(false);
    const [containsHeader, setContainsHeader] = useState(true);
    const [containsText, setContainsText] = useState(false);
    const [datasetId, setDatasetId] = useState(-1); // For loading/saving specific datasets

    // State for UI messages/logs
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Progress tracking state
    const [progress, setProgress] = useState({
        show: false,
        percentage: 0,
        operation: '',
        status: ''
    });

    // File upload state
    const [uploadedFiles, setUploadedFiles] = useState({});

    // State for connection status and error handling
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [isOffline, setIsOffline] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const [lastError, setLastError] = useState(null);
    const [showErrorDetails, setShowErrorDetails] = useState(false);

    // Helper function for UI messages
    const showMessage = (msg, type = 'info') => {
        setMessage({ text: msg, type });
        setTimeout(() => setMessage(''), 5000); // Clear message after 5 seconds
    };

    // Enhanced error handling with user feedback
    const handleApiError = (error, endpoint) => {
        console.error(`API Error for ${endpoint}:`, error);

        setLastError({
            message: error.message,
            code: error.code,
            suggestions: error.suggestions || [],
            endpoint,
            timestamp: error.timestamp || new Date().toISOString()
        });

        let errorType = 'error';
        let displayMessage = error.message;

        if (error.code === 'NETWORK_ERROR') {
            setConnectionStatus('disconnected');
            setIsOffline(true);
            errorType = 'warning';
        } else if (error.code === 'TIMEOUT_ERROR') {
            errorType = 'warning';
            displayMessage += ' (Request timed out)';
        } else if (error.code === 'SERVER_ERROR') {
            setConnectionStatus('error');
            errorType = 'error';
        }

        showMessage(displayMessage, errorType);
        setRetryCount(prev => prev + 1);
    };

    // Connection status monitoring
    useEffect(() => {
        // This would be set up when the client is initialized
        // For now, we'll simulate connection monitoring
        const checkConnection = async () => {
            try {
                // Attempt a simple health check
                const response = await fetch('http://localhost:8080/health', {
                    method: 'GET',
                    timeout: 5000
                });
                if (response.ok) {
                    setConnectionStatus('connected');
                    setIsOffline(false);
                } else {
                    setConnectionStatus('error');
                }
            } catch (error) {
                setConnectionStatus('disconnected');
                setIsOffline(true);
            }
        };

        checkConnection();
        const interval = setInterval(checkConnection, 30000); // Check every 30 seconds

        return () => clearInterval(interval);
    }, []);

    // Generic handler for form field changes
    const handleChange = (setter) => (e) => {
        setter(e.target.value);
    };

    // Handler for number input changes
    const handleNumberChange = (setter) => (e) => {
        setter(Number(e.target.value));
    };

    // Handler for checkbox/switch changes
    const handleCheckboxChange = (setter) => (e) => {
        setter(e.target.checked);
    };

    // Enhanced API calls with improved error handling and retry options
    const apiCall = async (endpoint, params = {}, retryAttempt = 0) => {
        setIsLoading(true);
        showMessage(`Executing ${endpoint}...`, 'info');

        try {
            const response = await fetch(`http://localhost:8080/${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    inputFile, targetFile, outputCsv, delimiter, numSamples, language, embeddingFile,
                    epochs, learningRate, layers, neurons, minRange, maxRange,
                    dbHost, dbPort, dbUser, dbPassword, dbSchema, sslMode, createTables,
                    containsHeader, containsText, datasetId,
                    ...params
                })
            });

            if (!response.ok) {
                const errorText = await response.text().catch(() => response.statusText);
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const result = await response.json();

            if (result.success) {
                showMessage(`${endpoint} completed successfully!`, 'success');
                setConnectionStatus('connected');
                setIsOffline(false);
                setRetryCount(0);
                setLastError(null);
            } else {
                const error = new Error(result.error || 'Unknown error');
                error.code = 'API_ERROR';
                throw error;
            }

            return result;
        } catch (error) {
            console.error(`API call failed:`, error);
            handleApiError(error, endpoint);

            // Auto-retry for certain errors
            if (retryAttempt < 2 && (error.message.includes('timeout') || error.message.includes('network'))) {
                const delay = Math.pow(2, retryAttempt) * 1000; // Exponential backoff
                showMessage(`Retrying ${endpoint} in ${delay/1000}s...`, 'warning');
                await new Promise(resolve => setTimeout(resolve, delay));
                return apiCall(endpoint, params, retryAttempt + 1);
            }

            return { success: false, error: error.message };
        } finally {
            setIsLoading(false);
        }
    };

    // Manual retry function
    const retryLastAction = async () => {
        if (lastError) {
            showMessage('Retrying last action...', 'info');
            // This would need to be implemented based on the last successful action
            // For now, just attempt a health check
            try {
                const response = await fetch('http://localhost:8080/health');
                if (response.ok) {
                    setConnectionStatus('connected');
                    setIsOffline(false);
                    showMessage('Connection restored!', 'success');
                    setLastError(null);
                }
            } catch (error) {
                showMessage('Retry failed. Please check server status.', 'error');
            }
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 p-6 font-inter antialiased rounded-lg shadow-lg">
            <script src="https://cdn.tailwindcss.com"></script>
            {/* Removed Lucide React script, using inline SVGs now */}
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />

            {/* Tailwind CSS config for Inter font */}
            <style>{`
        body { font-family: 'Inter', sans-serif; }
      `}</style>

            <div className="max-w-4xl mx-auto bg-gray-800 rounded-xl shadow-2xl p-8 space-y-8 border border-gray-700">
                <h1 className="text-4xl font-extrabold text-center text-blue-400 mb-10 tracking-wide">
                    CoreAI3D Configuration
                </h1>

                {/* Connection Status Indicator */}
                <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg shadow-md mb-4">
                    <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                            connectionStatus === 'connected' ? 'bg-green-500' :
                            connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                            connectionStatus === 'error' ? 'bg-red-500' : 'bg-gray-500'
                        }`}></div>
                        <span className="text-gray-300 font-medium">
                            Status: {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                            {isOffline && ' (Offline Mode)'}
                        </span>
                        {retryCount > 0 && (
                            <span className="text-yellow-400 text-sm">
                                (Retry #{retryCount})
                            </span>
                        )}
                    </div>
                    {(connectionStatus === 'disconnected' || connectionStatus === 'error') && (
                        <button
                            onClick={retryLastAction}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition duration-200 flex items-center space-x-2"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            <span>Retry</span>
                        </button>
                    )}
                </div>

                {isLoading && (
                    <div className="flex items-center justify-center p-4 bg-blue-600 rounded-lg shadow-md animate-pulse">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.0 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span className="text-white text-lg font-medium">{message.text}</span>
                    </div>
                )}
                {message.text && !isLoading && (
                    <div className={`p-4 rounded-lg shadow-md text-center ${
                        message.type === 'error' ? 'bg-red-600' :
                        message.type === 'warning' ? 'bg-yellow-600' :
                        message.type === 'success' ? 'bg-green-600' : 'bg-blue-600'
                    }`}>
                        <span className="text-white text-lg font-medium">{message.text}</span>
                    </div>
                )}

                {/* Error Details Panel */}
                {lastError && (
                    <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-4">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-red-300 font-semibold">Error Details</h3>
                            <button
                                onClick={() => setShowErrorDetails(!showErrorDetails)}
                                className="text-red-400 hover:text-red-300"
                            >
                                {showErrorDetails ? 'Hide' : 'Show'} Details
                            </button>
                        </div>
                        <p className="text-red-200 mb-2">{lastError.message}</p>
                        {showErrorDetails && (
                            <div className="text-sm text-red-300">
                                <p><strong>Code:</strong> {lastError.code}</p>
                                <p><strong>Endpoint:</strong> {lastError.endpoint}</p>
                                <p><strong>Time:</strong> {lastError.timestamp}</p>
                                {lastError.suggestions && lastError.suggestions.length > 0 && (
                                    <div className="mt-2">
                                        <strong>Suggestions:</strong>
                                        <ul className="list-disc list-inside mt-1">
                                            {lastError.suggestions.map((suggestion, index) => (
                                                <li key={index}>{suggestion}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* File and Data Loading Options */}
                <Section title="File & Data Options" icon="FileText" />
                <Section title="Training Parameters" icon="Brain" />
                <Section title="Database Configuration" icon="Database" />
                <Section title="Actions" icon="Play" />

                {/* The actual content of the sections */}
                <SectionContent>
                    <InputGroup label="Input File (--input-file)" value={inputFile} onChange={handleChange(setInputFile)} type="text" placeholder="e.g., data.csv" />
                    <InputGroup label="Target File (--target-file)" value={targetFile} onChange={handleChange(setTargetFile)} type="text" placeholder="Optional: targets.csv" />
                    <InputGroup label="Output CSV (--output-csv)" value={outputCsv} onChange={handleChange(setOutputCsv)} type="text" placeholder="e.g., results.csv" />
                    <InputGroup label="Delimiter (--delimiter)" value={delimiter} onChange={handleChange(setDelimiter)} type="text" maxLength={1} placeholder="e.g., ," />
                    <InputGroup label="Number of Samples (--num-samples)" value={numSamples} onChange={handleNumberChange(setNumSamples)} type="number" placeholder="-1 for all" />
                    <ToggleSwitch label="Contains Header (--contains-header)" checked={containsHeader} onChange={handleCheckboxChange(setContainsHeader)} />
                    <ToggleSwitch label="Contains Text (--contains-text)" checked={containsText} onChange={handleCheckboxChange(setContainsText)} />
                    <InputGroup label="Language Code (--language)" value={language} onChange={handleChange(setLanguage)} type="text" placeholder="e.g., en, nl" />
                    <InputGroup label="Embedding File (--embeding-file)" value={embeddingFile} onChange={handleChange(setEmbeddingFile)} type="text" placeholder="e.g., embeddings.txt" />
                </SectionContent>

                <SectionContent>
                    <InputGroup label="Epochs (--epochs)" value={epochs} onChange={handleNumberChange(setEpochs)} type="number" min="1" />
                    <InputGroup label="Learning Rate (--learning-rate)" value={learningRate} onChange={handleNumberChange(setLearningRate)} type="number" step="0.001" min="0" />
                    <InputGroup label="Layers (--layers)" value={layers} onChange={handleNumberChange(setLayers)} type="number" min="1" />
                    <InputGroup label="Neurons per Layer (--neurons)" value={neurons} onChange={handleNumberChange(setNeurons)} type="number" min="1" />
                    <InputGroup label="Min Range (--min)" value={minRange} onChange={handleNumberChange(setMinRange)} type="number" step="0.1" />
                    <InputGroup label="Max Range (--max)" value={maxRange} onChange={handleNumberChange(setMaxRange)} type="number" step="0.1" />
                </SectionContent>

                <SectionContent>
                    <InputGroup label="DB Host (--db-host)" value={dbHost} onChange={handleChange(setDbHost)} type="text" placeholder="e.g., 0.0.0.0" />
                    <InputGroup label="DB Port (--db-port)" value={dbPort} onChange={handleNumberChange(setDbPort)} type="number" />
                    <InputGroup label="DB User (--db-user)" value={dbUser} onChange={handleChange(setDbUser)} type="text" />
                    <InputGroup label="DB Password (--db-password)" value={dbPassword} onChange={handleChange(setDbPassword)} type="password" />
                    <InputGroup label="DB Schema (--db-schema)" value={dbSchema} onChange={handleChange(setDbSchema)} type="text" />
                    <SelectGroup label="SSL Mode (--ssl-mode)" value={sslMode} onChange={handleChange(setSslMode)} options={['DISABLED', 'VERIFY_CA', 'VERIFY_IDENTITY']} />
                    <ToggleSwitch label="Create Tables (--create-tables)" checked={createTables} onChange={handleCheckboxChange(setCreateTables)} />
                    <InputGroup label="Dataset ID (for DB operations)" value={datasetId} onChange={handleNumberChange(setDatasetId)} type="number" placeholder="-1 for new/latest" />
                </SectionContent>

                <SectionContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <ActionButton onClick={() => apiCall('load-csv')} disabled={isLoading} label="Load CSV" />
                        <ActionButton onClick={() => apiCall('load-targets-csv')} disabled={isLoading} label="Load Targets CSV" />
                        <ActionButton onClick={() => apiCall('load-dataset-db')} disabled={isLoading || datasetId === -1} label="Load Dataset from DB" />
                        <ActionButton onClick={() => apiCall('preprocess')} disabled={isLoading} label="Preprocess Data" />
                        <ActionButton onClick={() => apiCall('train')} disabled={isLoading} label="Train Model" />
                        <ActionButton onClick={() => apiCall('calculate-rmse')} disabled={isLoading} label="Calculate RMSE" />
                        <ActionButton onClick={() => apiCall('save-model')} disabled={isLoading || datasetId === -1} label="Save Model" />
                        <ActionButton onClick={() => apiCall('load-model')} disabled={isLoading || datasetId === -1} label="Load Model" />
                        <ActionButton onClick={() => apiCall('save-results-csv')} disabled={isLoading} label="Save Results to CSV" />
                    </div>
                </SectionContent>
            </div>
        </div>
    );
};

// Section Component for grouping related inputs
const Section = ({ title, icon }) => {
    // Define simple SVG icons here
    const icons = {
        FileText: (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-blue-400">
                <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"></path>
                <path d="M14 2v4a2 2 0 0 0 2 2h4"></path>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <line x1="10" y1="9" x2="8" y2="9"></line>
            </svg>
        ),
        Brain: (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-blue-400">
                <path d="M12 2a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"></path>
                <path d="M12 2v10a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"></path>
                <path d="M22 10V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2z"></path>
                <path d="M22 14v4a2 2 0 0 0-2 2H4a2 2 0 0 0-2-2v-4a2 2 0 0 0 2-2h16a2 2 0 0 0 2-2z"></path>
            </svg>
        ),
        Database: (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-blue-400">
                <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                <path d="M3 12a9 3 0 0 0 18 0"></path>
                <path d="M3 19a9 3 0 0 0 18 0"></path>
                <path d="M3 5v14"></path>
                <path d="M21 5v14"></path>
            </svg>
        ),
        Play: (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-blue-400">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
        )
    };
    const Icon = icons[icon]; // Get the SVG element based on the icon name

    return (
        <div className="bg-gray-700 p-6 rounded-lg shadow-inner border border-gray-600">
            <h2 className="text-2xl font-semibold text-blue-300 mb-6 flex items-center">
                {Icon && <span className="mr-3">{Icon}</span>} {/* Render SVG directly */}
                {title}
            </h2>
        </div>
    );
};

// Section Content Wrapper - This component now holds the actual inputs
const SectionContent = ({ children }) => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-y-4 gap-x-6">
        {children}
    </div>
);

// Input Group Component (Text, Number, Password)
const InputGroup = ({ label, value, onChange, type = 'text', placeholder = '', min, step, maxLength }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
        <input
            type={type}
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            min={min}
            step={step}
            maxLength={maxLength}
            className="w-full p-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:ring-blue-500 focus:border-blue-500 transition duration-200 shadow-sm"
        />
    </div>
);

// Toggle Switch Component
const ToggleSwitch = ({ label, checked, onChange }) => (
    <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-300 cursor-pointer">{label}</label>
        <label htmlFor={label.replace(/\s/g, '-').toLowerCase()} className="relative inline-flex items-center cursor-pointer">
            <input
                type="checkbox"
                id={label.replace(/\s/g, '-').toLowerCase()}
                className="sr-only peer"
                checked={checked}
                onChange={onChange}
            />
            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:border-gray-300 after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
        </label>
    </div>
);

// Select Group Component (Dropdown)
const SelectGroup = ({ label, value, onChange, options }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
        <select
            value={value}
            onChange={onChange}
            className="w-full p-2 bg-gray-900 border border-gray-600 rounded-md text-white focus:ring-blue-500 focus:border-blue-500 transition duration-200 shadow-sm appearance-none pr-8"
            style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='1.5' stroke='currentColor' class='w-6 h-6'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M8.25 15L12 18.75 15.75 15m-7.5-6L12 5.25 15.75 9'/%3E%3C/svg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.5rem center',
                backgroundSize: '1.5em'
            }}
        >
            {options.map(option => (
                <option key={option} value={option}>
                    {option}
                </option>
            ))}
        </select>
    </div>
);

// Action Button Component
const ActionButton = ({ onClick, disabled, label }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`w-full py-3 px-6 rounded-lg text-lg font-bold transition duration-300 ease-in-out shadow-md
      ${disabled
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white transform hover:scale-105 active:scale-95'
            }`}
    >
        {label}
    </button>
);

export default App;

