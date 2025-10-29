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

    // Helper function for UI messages
    const showMessage = (msg, type = 'info') => {
        setMessage({ text: msg, type });
        setTimeout(() => setMessage(''), 5000); // Clear message after 5 seconds
    };

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

    // Simulate API calls (replace with actual fetch to your C++ backend)
    const simulateApiCall = async (actionName) => {
        setIsLoading(true);
        showMessage(`Executing ${actionName}...`, 'info');
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        setIsLoading(false);
        showMessage(`${actionName} completed successfully! (Simulated)`, 'success');
        console.log(`Action: ${actionName}`, {
            inputFile, targetFile, outputCsv, delimiter, numSamples, language, embeddingFile,
            epochs, learningRate, layers, neurons, minRange, maxRange,
            dbHost, dbPort, dbUser, dbPassword, dbSchema, sslMode, createTables,
            containsHeader, containsText, datasetId
        });
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
                    <div className={`p-4 rounded-lg shadow-md text-center ${message.type === 'info' ? 'bg-blue-600' : 'bg-green-600'}`}>
                        <span className="text-white text-lg font-medium">{message.text}</span>
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
                        <ActionButton onClick={() => simulateApiCall('Load CSV')} disabled={isLoading} label="Load CSV" />
                        <ActionButton onClick={() => simulateApiCall('Load Targets CSV')} disabled={isLoading} label="Load Targets CSV" />
                        <ActionButton onClick={() => simulateApiCall('Load Dataset from DB')} disabled={isLoading || datasetId === -1} label="Load Dataset from DB" />
                        <ActionButton onClick={() => simulateApiCall('Preprocess Data')} disabled={isLoading} label="Preprocess Data" />
                        <ActionButton onClick={() => simulateApiCall('Train Model')} disabled={isLoading} label="Train Model" />
                        <ActionButton onClick={() => simulateApiCall('Calculate RMSE')} disabled={isLoading} label="Calculate RMSE" />
                        <ActionButton onClick={() => simulateApiCall('Save Model')} disabled={isLoading || datasetId === -1} label="Save Model" />
                        <ActionButton onClick={() => simulateApiCall('Load Model')} disabled={isLoading || datasetId === -1} label="Load Model" />
                        <ActionButton onClick={() => simulateApiCall('Save Results to CSV')} disabled={isLoading} label="Save Results to CSV" />
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

