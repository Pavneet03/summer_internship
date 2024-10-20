import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.signal import butter, lfilter

# Load the seizure events data
events_data = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\Documents\internship\seizure_events.csv')

# Step 1: Data Preprocessing

def load_eeg_data(series_id):
    eeg_file_path = rf'C:\Users\hp\OneDrive\Desktop\Documents\internship\seizure_events\{series_id}.edf'
    if not os.path.exists(eeg_file_path):
        print(f'File not found: {eeg_file_path}')
        return None
    eeg_data = pd.read_csv(eeg_file_path)
    return eeg_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=256, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_eeg(eeg_data):
    # Apply bandpass filter
    filtered_data = bandpass_filter(eeg_data['signal'], lowcut=0.5, highcut=40, fs=256)
    eeg_data['filtered_signal'] = filtered_data
    
    # Standardize the data
    scaler = StandardScaler()
    eeg_data['scaled_signal'] = scaler.fit_transform(eeg_data[['filtered_signal']])
    return eeg_data

# Step 2: Label Data Based on Seizure Timings
def label_data(eeg_data, onset, offset):
    eeg_data['label'] = 0  # Default label (no seizure)
    eeg_data.loc[(eeg_data['timestamp'] >= onset) & (eeg_data['timestamp'] <= offset), 'label'] = 1  # Ictal phase
    return eeg_data

# Step 3: Feature Extraction
def extract_features(eeg_data, window_size=256):
    features = []
    labels = []
    for start in range(0, len(eeg_data) - window_size, window_size):
        segment = eeg_data.iloc[start:start + window_size]
        mean_val = segment['scaled_signal'].mean()
        std_val = segment['scaled_signal'].std()
        max_val = segment['scaled_signal'].max()
        min_val = segment['scaled_signal'].min()
        label = segment['label'].mode()[0]  # Use the most frequent label in the window
        features.append([mean_val, std_val, max_val, min_val])
        labels.append(label)
    return np.array(features), np.array(labels)

# Step 4: Model Training
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')
    return accuracy

# Example Usage
all_features = []
all_labels = []

# Process each series in events_data
for index, row in events_data.iterrows():
    series_id = row['series_id']
    onset = row['onset']
    offset = row['offset']
    
    # Load and preprocess EEG data for the given series
    eeg_data = load_eeg_data(series_id)
    
    # Skip processing if the file is not found
    if eeg_data is None:
        continue
    
    eeg_data = preprocess_eeg(eeg_data)
    
    # Label the data
    eeg_data = label_data(eeg_data, onset, offset)
    
    # Extract features
    features, labels = extract_features(eeg_data)
    
    if features is not None and labels is not None:
        all_features.append(features)
        all_labels.append(labels)

# Convert to numpy arrays after checking if they are populated
if len(all_features) == 0 or len(all_labels) == 0:
    print("No features or labels found. Check the data processing logic.")
else:
    # Convert lists to numpy arrays
    all_features = np.concatenate(all_features)  # Ensure it's a numpy array
    all_labels = np.concatenate(all_labels)

# Ensure that the arrays have been created successfully
if isinstance(all_features, np.ndarray):
    # Print shapes for debugging
    print(f"Shape of all_features before reshape: {all_features.shape}")
else:
    print("all_features is not a NumPy array.")

if isinstance(all_labels, np.ndarray):
    print(f"Shape of all_labels: {all_labels.shape}")
else:
    print("all_labels is not a NumPy array.")

# Ensure all_features is 2D
if len(all_features.shape) == 1:
    all_features = all_features.reshape(-1, 1)  # Reshape to 2D if it's 1D

# Reshape for LSTM: [samples, time steps, features]
X = all_features.reshape(all_features.shape[0], 1, all_features.shape[1])
y = all_labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)
