import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import argparse
import shutil
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sys

def plot_anomalies(results_df, values_df, save_path=None):
    """
    Plot the anomalous intervals along with the original values and highlight labeled anomalies.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing detected anomaly intervals.
    - values_df (pd.DataFrame): DataFrame containing original values, timestamps, and labels.
    - save_path (str): Path to save the plots. If None, plots will be displayed.
    """
    for idx, anomaly in results_df.iterrows():
        chan_id = anomaly['chan_id']
        start, end = eval(anomaly['anomaly_sequences'])  # Convert string to list
        
        # Filter the values for this channel and time range
        channel_data = values_df[(values_df['chan_id'] == chan_id) & 
                                 (values_df['timestamp'] >= start - 400) & 
                                 (values_df['timestamp'] <= end + 400)]
        
        # Plot the data
        plt.figure(figsize=(14, 7))
        plt.plot(channel_data['timestamp'], channel_data['value'], label='Original Values')
        plt.axvspan(start, end, color='red', alpha=0.3, label='Detected Anomalous Interval')
        
        # Highlight labeled anomalies
        labeled_anomalies = channel_data[channel_data['label'] == 1]
        if not labeled_anomalies.empty:
            plt.fill_between(labeled_anomalies['timestamp'], 
                             labeled_anomalies['value'].min(), 
                             labeled_anomalies['value'].max(), 
                             color='yellow', alpha=0.3, label='Labeled Anomalous Interval')
        
        plt.title(f'Anomalous Interval for {chan_id} [{start}, {end}]')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f"{chan_id}_{start}_{end}.png"))
        else:
            plt.show()

# Combine training and testing scores, ensuring the correct indexing
def combine_train_test_scores(train_scores_df, test_scores_df):
    combined_dfs = []
    train_lengths = {}
    test_lengths = {}

    for signal_idx, group in test_scores_df.groupby('signal_idx'):
        train_group = train_scores_df[train_scores_df['signal_idx'] == signal_idx]
        combined_group = pd.concat([train_group, group], ignore_index=True)

        # Store lengths of training and testing portions
        train_lengths[signal_idx] = len(train_group)
        test_lengths[signal_idx] = len(group)

        combined_dfs.append(combined_group)
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    return combined_df, train_lengths, test_lengths


def calculate_likelihood_chunk(chunk, lw=2100, lshort=0, threshold=0.99):
    likelihood = []
    anomalies = []

    chunk = chunk.reset_index(drop=True)
    for idx, row in chunk.iterrows():
        t = idx
        score = row['scores']
        # Update window Win
        if t < lw:
            Win = chunk['scores'][:t+1]
        else:
            Win = chunk['scores'][t-lw+1:t+1]
        
        # Calculate mean and variance for Win
        mu_Win = np.mean(Win)
        sigma_Win = np.var(Win)
        
        # Update short-term average
        if t < lshort:
            mu_s = mu_Win
        else:
            short_window = chunk['scores'][t-lshort:t+1]
            mu_s = np.mean(short_window)
        
        # Calculate likelihood of anomaly
        if sigma_Win > 0:
            lik = 1 - norm.cdf((mu_s - mu_Win) / np.sqrt(sigma_Win))
        else:
            lik = 0  # If variance is zero, the likelihood is set to 0
        
        # Mark as anomaly if likelihood exceeds threshold
        is_anomalous = 1 if lik >= threshold else 0
        
        # Store the results
        likelihood.append(lik)
        anomalies.append(is_anomalous)

    # Add the results to the chunk
    chunk['likelihood'] = likelihood
    threshold_name = f'{threshold}'.replace('.', '_')
    chunk[f'anomalous_{threshold_name}'] = anomalies

    return chunk

def calculate_likelihood(scores_df, lw=2100, lshort=0, threshold=0.99):
    # Automatically detect the number of available cores
    n_cores = cpu_count()
    print(f"Using {n_cores} cores")

    # Group the DataFrame by 'signal_idx'
    grouped = scores_df.groupby('signal_idx')

    # Use multiprocessing to process each group in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.starmap(calculate_likelihood_chunk, [(group, lw, lshort, threshold) for name, group in grouped])

    # Combine the results back into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)

    return final_df

def track_anomalous_intervals(scores_df, threshold_name='0_99', min_duration=30, global_threshold=None):
    detection_results = []
    current_interval = None
    
    for idx, row in scores_df.iterrows():
        # Check if the point is anomalous based on the original threshold and the global threshold
        is_anomalous = (row[f'anomalous_{threshold_name}'] == 1)

        
        if is_anomalous:
            if current_interval is None:
                # Initialize an interval
                current_interval = {
                    'chan_id': row['signal_idx'],
                    'start': row['timestamp'],
                    'scores': [],
                    'likelihoods': []
                }
            current_interval['end'] = row['timestamp']
            current_interval['scores'].append(row['scores'])
            current_interval['likelihoods'].append(row['likelihood'])
        else:
            if current_interval is not None:
                # Finalize the current interval
                avg_score = np.mean(current_interval['scores'])
                avg_likelihood = np.mean(current_interval['likelihoods'])
                if current_interval['end'] - current_interval['start'] >= min_duration:
                    if global_threshold is None or avg_score > global_threshold: # Global thresholding                   
                        detection_results.append({
                            'chan_id': current_interval['chan_id'],
                            'anomaly_sequences': f"[{current_interval['start']}, {current_interval['end']}]",
                            'score': avg_score,
                            'likelihood': avg_likelihood,
                            'hard_threshold': global_threshold
                        })
                current_interval = None

    # Convert the detection results into a DataFrame
    results_df = pd.DataFrame(detection_results)
    
    return results_df


def check_overlap(detected_sequence, labeled_sequence, buffer=300):
    """Check if the detected anomaly overlaps with the labeled anomaly or falls within the buffer range."""
    detected_start, detected_end = detected_sequence
    labeled_start, labeled_end = labeled_sequence
    
    # Check if there is an overlap or the detected anomaly is within 300 points before the labeled anomaly
    return not (detected_end < labeled_start - buffer or detected_start > labeled_end)

def evaluate_anomalies(results_df, labeled_anomalies):
    # Initialize counters
    overall_TP, overall_FP, overall_FN, collective_FN, collective_TP = 0, 0, 0, 0, 0
    tp_by_spacecraft = {'SMAP': 0, 'MSL': 0}
    tp_by_class = {class_label: 0 for class_label in labeled_anomalies['class'].unique()}

    # Convert string representation of lists to actual lists
    results_df['anomaly_sequences'] = results_df['anomaly_sequences'].apply(eval)
    labeled_anomalies['anomaly_sequences'] = labeled_anomalies['anomaly_sequences'].apply(eval)
    
    # Iterate through results to classify TPs, FPs, FNs
    for index, result_row in results_df.iterrows():
        detected_sequence = result_row['anomaly_sequences']
        detected_chan_id = result_row['chan_id']
        
        # Find labeled anomalies for the same channel
        matching_labeled = labeled_anomalies[labeled_anomalies['chan_id'] == detected_chan_id]
        
        # Initialize flags
        is_TP = False
        for _, labeled_row in matching_labeled.iterrows():
            labeled_sequence = labeled_row['anomaly_sequences']
            spacecraft = labeled_row['spacecraft']
            anomaly_class = labeled_row['class']
            
            # Check for overlap or proximity
            if check_overlap(detected_sequence, labeled_sequence):
                is_TP = True
                overall_TP += 1
                tp_by_spacecraft[spacecraft] += 1
                tp_by_class[anomaly_class] += 1
                break
        
        if not is_TP:
            overall_FP += 1
        

    
    # Count FNs (labeled anomalies that were not detected)
    for _, labeled_row in labeled_anomalies.iterrows():
        labeled_sequence = labeled_row['anomaly_sequences']
        labeled_chan_id = labeled_row['chan_id']
        
        matching_detected = results_df[results_df['chan_id'] == labeled_chan_id]
        is_FN = True
        for _, result_row in matching_detected.iterrows():
            detected_sequence = result_row['anomaly_sequences']
            if check_overlap(detected_sequence, labeled_sequence):
                is_FN = False
                break
        
        if is_FN:
            if labeled_row['class'] != "Neglected":
                collective_FN += 1
            overall_FN += 1

    collective_TP = overall_TP - tp_by_class.get("Neglected", 0)
    collective_FP = overall_FP + tp_by_class.get("Neglected", 0)

    # Calculate precision, recall, and F0.5 score
    precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0
    recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0
    collective_precision = collective_TP / (collective_TP + collective_FP) if (collective_TP + collective_FP) > 0 else 0
    collective_recall = collective_TP / (collective_TP + collective_FN) if (collective_TP + collective_FN) > 0 else 0
    f0_5 = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall) if (precision + recall) > 0 else 0
    collectivef0_5 = (1 + 0.5**2) * (collective_precision * collective_recall) / (0.5**2 * collective_precision + collective_recall) if (collective_precision + collective_recall) > 0 else 0
    
    # Prepare the result metrics
    result_metrics = {
        'Total TP': [overall_TP],
        'Total FP': [overall_FP],
        'Total FN': [overall_FN],
        'Collective TP': [collective_TP],
        'Collective FN': [collective_FN],
        'Precision': [precision],
        'Collective Precision': [collective_precision],
        'Recall': [recall],
        'F0.5 Score': [f0_5],
        'Collective F0.5 Score': [collectivef0_5],
        'TP in SMAP': [tp_by_spacecraft.get('SMAP', 0)],
        'TP in MSL': [tp_by_spacecraft.get('MSL', 0)],
    }

    # Add TP by class to result_metrics
    for anomaly_class, count in tp_by_class.items():
        result_metrics[f'TP in {anomaly_class}'] = [count]

    # Convert result_metrics to DataFrame
    result_metrics_df = pd.DataFrame(result_metrics)

    return result_metrics_df



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection method.')
    parser.add_argument('--method', type=str, required=True, help='Method to evaluate (e.g., SAD, AE)')
    parser.add_argument('--plot', action='store_true', help='Flag to plot anomalies')
    parser.add_argument('--likelihood_threshold', type=float, default=0.99, help='Likelihood threshold quantile for labeling anomalies (default: 0.99)')
    parser.add_argument('--score_threshold', type=float, default=0.25, help='Score threhold quantile for defining the global threshold (default: 0.25)')

    args = parser.parse_args()

    # Method to evaluate
    method = args.method
    plot_anomalies_flag = args.plot
    likelihood_threshold = args.likelihood_threshold
    score_threshold = args.score_threshold
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../results/{method}') 
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data') 

    print(f"Evaluation for method {method} started, the likelihood quantile threshold is set to be q{likelihood_threshold}")

    # Load the training and testing scores
    train_scores_df = pd.read_csv(os.path.join(result_path, 'detection_train/idx_score.csv'))
    test_scores_df = pd.read_csv(os.path.join(result_path, 'detection/idx_score.csv'))
    labeled_anomalies = pd.read_csv(os.path.join(data_path, 'raw/relabeled_anomalies.csv'))

    # Calculate the global threshold based on the training scores
    global_threshold = np.percentile(train_scores_df['scores'], score_threshold*100, axis=0)

    print(f"The score threshold quantile is set to be {global_threshold}")

    # Combine the training and testing scores
    combined_scores_df, train_lengths, test_lengths = combine_train_test_scores(train_scores_df, test_scores_df)

    # Calculate likelihood and label anomalies
    combined_scores_df = calculate_likelihood(combined_scores_df, lw=2100, lshort=0, threshold=likelihood_threshold)

    # Separate the testing data based on the lengths tracked earlier
    results = []
    for signal_idx, group in combined_scores_df.groupby('signal_idx'):
        test_start_idx = train_lengths[signal_idx]
        test_end_idx = test_start_idx + test_lengths[signal_idx]
        test_group = group.iloc[test_start_idx:test_end_idx].copy()
        results.append(test_group)

    test_scores_df = pd.concat(results, ignore_index=True)

    # Save the likelihoods for the testing portion
    test_scores_df.to_csv(os.path.join(result_path, 'detection/idx_score_likelihood.csv'), index=False)

    # Track anomalous intervals in the testing data using the new global threshold rule
    threshold_name = f'{likelihood_threshold}'.replace('.', '_')  # Replace '.' with '_' for the filename
    results_df = track_anomalous_intervals(test_scores_df, threshold_name=threshold_name, min_duration=30, global_threshold=global_threshold)
    results_df.to_csv(os.path.join(result_path, 'detection/anomaly_intervals.csv'), index=False)

    # Call the evaluation function
    evaluation_metrics = evaluate_anomalies(results_df.copy(deep=True), labeled_anomalies)
    evaluation_metrics.to_csv(os.path.join(result_path, 'detection/metrics.csv'), index=False)

    # Examine the detected anomalies
    if plot_anomalies_flag:
        plots_dir = os.path.join(result_path, 'plots')
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir)  # Reset the folder
        os.makedirs(plots_dir, exist_ok=True)
        values_df = pd.read_csv(os.path.join(data_path, 'processed/labeled_test_data.csv'))
        plot_anomalies(results_df.copy(deep=True), values_df, save_path=plots_dir)

    print(f"Evaluation for method {method} ended")