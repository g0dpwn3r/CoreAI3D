CREATE database CoreAI3D;

use CoreAI3D;

show tables;
CREATE TABLE datasets (
    dataset_id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL UNIQUE COMMENT 'Unique name for the dataset (e.g., historical_stock_prices, sensor_readings)',
    description TEXT COMMENT 'Detailed description of the dataset, e.g., source, collection method',
    num_rows INT COMMENT 'Total number of rows/samples in the dataset',
    num_features INT COMMENT 'Number of input features/columns for this dataset',
    num_labels INT COMMENT 'Number of output labels/target columns for this dataset',
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the dataset metadata was added'
);



CREATE TABLE dataset_records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL COMMENT 'Foreign Key linking to the datasets table',
    row_index INT NOT NULL COMMENT 'The original row number of this record in the CSV file (0-indexed or 1-indexed)',
    feature_values JSON COMMENT 'JSON array of input feature values for this row (e.g., [val1, val2, ...])',
    label_values JSON COMMENT 'JSON array of output label/target values for this row (e.g., [label1, label2, ...])',
    
    -- Add a unique constraint to ensure each row_index within a dataset_id is unique
    UNIQUE (dataset_id, row_index),

    -- Define the foreign key constraint
    CONSTRAINT fk_dataset_record
        FOREIGN KEY (dataset_id)
        REFERENCES datasets(dataset_id)
        ON DELETE CASCADE -- If a dataset record is deleted, all its associated raw data records are also deleted
);


CREATE TABLE ai_model_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when this model state was saved',
    dataset_id INT COMMENT 'Foreign Key referencing the datasets table, indicating which dataset the model was trained on',
    input_data LONGBLOB COMMENT 'Serialized input data matrix (internal model state)',
    output_data LONGBLOB COMMENT 'Serialized output data matrix (internal model state)',
    hidden_data LONGBLOB COMMENT 'Serialized hidden layer activations (internal model state)',
    hidden_output_data LONGBLOB COMMENT 'Serialized hidden layer output (internal model state)',
    hidden_error_data LONGBLOB COMMENT 'Serialized hidden layer error (internal model state)',
    weights_hidden_input LONGBLOB COMMENT 'Serialized weights matrix connecting input to hidden layer',
    weights_output_hidden LONGBLOB COMMENT 'Serialized weights matrix connecting hidden to output layer',
    
    -- Define the foreign key constraint
    CONSTRAINT fk_model_state_dataset
        FOREIGN KEY (dataset_id)
        REFERENCES datasets(dataset_id)
        ON DELETE SET NULL -- If a dataset is deleted, set dataset_id in ai_model_states to NULL
);

CREATE TABLE IF NOT EXISTS prediction_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL COMMENT 'Foreign Key linking to the datasets table',
    sample_index INT NOT NULL COMMENT 'The index of the sample within the dataset',
    input_features JSON COMMENT 'JSON array of input feature values',
    actual_targets JSON COMMENT 'JSON array of actual target values',
    predicted_targets JSON COMMENT 'JSON array of predicted target values',
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the prediction was saved',
    UNIQUE (dataset_id, sample_index),
    CONSTRAINT fk_prediction_dataset
        FOREIGN KEY (dataset_id)
        REFERENCES datasets(dataset_id)
        ON DELETE CASCADE
);

DROP TABLE IF EXISTS predictions;

show tables;
