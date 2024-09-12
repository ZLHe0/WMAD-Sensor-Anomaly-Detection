import pandas as pd
import numpy as np
import os
import ast

def median_mad_sigmoid_normalize(signal_df, channel, apply_existing=False):
    """Apply Median-MAD-Sigmoid normalization to a signal DataFrame."""
    if apply_existing:
        median = normalization_factor[channel]['median']
        mad = normalization_factor[channel]['mad']
    else:
        median = signal_df['value'].median()
        mad = np.mean(np.abs(signal_df['value'] - median))
        if mad == 0:
            mad = 1  # To prevent division by zero
        
        # Store normalization info
        normalization_factor[channel] = {'median': median, 'mad': mad}
    
    normalized = (signal_df['value'] - median) / mad
    
    # Apply modified sigmoid function to map to [-1, 1]
    signal_df['value'] = 2 / (1 + np.exp(-normalized)) - 1
    
    return signal_df

def label_data(signal_df, events, pattern=None):
    start, end = events
    # Label the data points within the event window as 1
    signal_df.loc[(signal_df['timestamp'] >= start) & (signal_df['timestamp'] <= end), 'label'] = 1
    signal_df.loc[(signal_df['timestamp'] >= start) & (signal_df['timestamp'] <= end), 'pattern'] = pattern
    return signal_df

def process_train_data():
    output_df = pd.DataFrame()
    channels_used_in_train = []

    # Iterate through all files in the train folder
    for file_name in os.listdir(os.path.join(data_path, 'train')):
        if file_name.endswith('.npy'):
            signal = os.path.splitext(file_name)[0]  # Get the signal name (without .npy extension)
            
            # Load the signal data from the training folder
            file_path = os.path.join(data_path, 'train', file_name)
            data = np.load(file_path)
            signal_df = pd.DataFrame(data[:, 0], columns=['value'])
            signal_df['timestamp'] = range(len(signal_df))  # Assuming time is just the index here
            
            # Initialize all labels as 0 (normal)
            signal_df['label'] = 0
            signal_df['pattern'] = None
            
            # Check if this signal is in the training_labels_df
            training_signal_rows = training_labels_df[training_labels_df['chan_id'] == signal]
            if not training_signal_rows.empty:
                # Mark this channel as used in training data
                channels_used_in_train.append(signal)
                
                # Load the corresponding test data
                test_file_path = os.path.join(data_path, 'test', file_name)
                test_data = np.load(test_file_path)
                test_data = pd.DataFrame(test_data[:, 0], columns=['value'])
                
                # Reset timestamps for the test data
                test_data['timestamp'] = range(len(test_data))
                test_data['label'] = 0
                test_data['pattern'] = None
                
                # Label the data for the specified anomalies
                for _, row in training_signal_rows.iterrows():
                    events = row['anomaly_sequences']
                    test_data = label_data(test_data, events, pattern=row['class'])
                
                # Concatenate the normal training data with the labeled test data
                signal_df = pd.concat([signal_df, test_data], ignore_index=True)
                signal_df['timestamp'] = range(len(signal_df))  # Reset the timestamp after concatenation
            
            # Normalize the concatenated signal data (training + test)
            signal_df = median_mad_sigmoid_normalize(signal_df, channel=signal)
            
            # Add the signal name to the DataFrame
            signal_df['chan_id'] = signal
            
            # Append to the overall DataFrame
            output_df = pd.concat([output_df, signal_df], ignore_index=True)
    
    # Save the normalization information
    normalization_factor_df = pd.DataFrame.from_dict(normalization_factor, orient='index')
    normalization_factor_df.to_csv(normalization_factor_path, index_label='chan_id')
    
    return output_df, channels_used_in_train

def process_test_data(channels_to_exclude):
    output_df = pd.DataFrame()

    # Load the normalization information
    normalization_factor_df = pd.read_csv(normalization_factor_path, index_col='chan_id')
    normalization_factor.update(normalization_factor_df.to_dict(orient='index'))

    # Iterate through all files in the test folder
    for file_name in os.listdir(os.path.join(data_path, 'test')):
        if file_name.endswith('.npy'):
            signal = os.path.splitext(file_name)[0]  # Get the signal name (without .npy extension)
            
            # Skip channels used in training data
            if signal in channels_to_exclude:
                continue
            
            # Load the signal data
            file_path = os.path.join(data_path, 'test', file_name)
            data = np.load(file_path)
            signal_df = pd.DataFrame(data[:, 0], columns=['value'])
            signal_df['timestamp'] = range(len(signal_df))  # Assuming time is just the index here
            
            # Normalize the signal data using training normalization info
            signal_df = median_mad_sigmoid_normalize(signal_df, channel=signal, apply_existing=True)
            
            # Initialize all labels as 0 (normal)
            signal_df['label'] = 0
            signal_df['pattern'] = None
            
            # Filter the labels_df for this particular signal
            signal_rows = labels_df[labels_df['chan_id'] == signal]
            
            # Label the data for each anomaly on this signal
            for _, row in signal_rows.iterrows():
                events = row['anomaly_sequences']
                signal_df = label_data(signal_df, events, pattern=row['class'])
            
            # Add the signal name to the DataFrame
            signal_df['chan_id'] = signal
            
            # Append to the overall DataFrame
            output_df = pd.concat([output_df, signal_df], ignore_index=True)
        
    return output_df



if __name__ == "__main__":
    # Paths to your data files
    training_labels_path = 'data/raw/training_labels.csv'
    full_labels_path = 'data/raw/relabeled_anomalies.csv'
    data_path = 'data/raw'
    normalization_factor_path = 'data/processed/normalization_factor.csv'

    # Load the specific training labels
    training_labels_df = pd.read_csv(training_labels_path)
    training_labels_df['anomaly_sequences'] = training_labels_df['anomaly_sequences'].apply(ast.literal_eval)

    # Load the full anomaly labels file
    labels_df = pd.read_csv(full_labels_path)
    labels_df['anomaly_sequences'] = labels_df['anomaly_sequences'].apply(ast.literal_eval)

    # Dictionary to store normalization info for each channel
    normalization_factor = {}

    # Process the training data
    train_df, channels_used_in_train = process_train_data()
    train_df.to_csv('data/processed/labeled_train_data.csv', index=False)
    print("Training data successfully processed")

    # Process the testing data, excluding channels used in training
    test_df = process_test_data(channels_used_in_train)
    test_df.to_csv('data/processed/labeled_test_data.csv', index=False)
    print("Testing data successfully processed")