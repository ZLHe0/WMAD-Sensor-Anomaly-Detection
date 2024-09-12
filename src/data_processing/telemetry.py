import sys
import os
import numpy as np
import pandas as pd
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset
import logging
import pdb
import tensorflow as tf
from networks.GTT_main.src.core.model import GTT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Telemetry_Dataset():
    """
    This class represents a dataset of telemetry time series data for anomaly detection. It is designed to handle both
    training and testing modes. In the training mode, it samples labeled and unlabeled time windows, preparing them for 
    the learning task. In the testing mode, it processes the entire time series into sliding windows, preparing 
    them for evaluation.
    """
    def __init__(self, data_path, mode, window_size, n_shots = 20, n_queries = 30, n_total_tasks = 2000, 
                  anomaly_shot_prop = 0.1, anomaly_sample_prop = 0.1, load_from_pretrain = False, foundation_path = None):
        self.signal_value_dict = {}
        self.signal_label_dict = {}
        self.window_label_dict = {}
        self.n_total_tasks = n_total_tasks
        self.anomaly_sample_prop = anomaly_sample_prop
        self.tasks = []
        self.mode = mode
        self.n_shots = n_shots
        self.n_queries = n_queries

        df_tel = pd.read_csv(data_path)
        for signal_idx in df_tel['chan_id'].unique():
            signal_value = np.array(df_tel.loc[df_tel['chan_id'] == signal_idx, 'value'])
            signal_label = np.array(df_tel.loc[df_tel['chan_id'] == signal_idx, 'label'])
            self.signal_value_dict[signal_idx] = signal_value
            self.window_label_dict[signal_idx] = signal_label
            self.signal_label_dict[signal_idx] = 1 if np.any(signal_label) == 1 else 0
        
        if self.mode == 'train' and len(self.signal_label_dict) > 0:
            sampled_tasks_indices = self.sample_signal_indices()
            for task_idx in sampled_tasks_indices:
                # get the data for the sampled task
                data_array = self.signal_value_dict[task_idx]
                label_array = self.window_label_dict[task_idx]
                if self.signal_label_dict[task_idx] == 1:
                    # Sampling for anomalous time series
                    if np.any(label_array == 1):
                        if anomaly_shot_prop * self.n_shots < 1:
                            logging.info("No labeled data will be sampled due to a small shot_prop")
                    
                    # 1. Collect valid labeled anomalies starting point
                    labeled_indices = np.where(label_array == 1)[0]
                    labeled_start = labeled_indices[0]; labeled_end = labeled_indices[-1] + 1
                    labeled_indices = labeled_indices[labeled_indices < labeled_end - window_size]
                    if labeled_indices.size == 0:
                        continue

                    # 2. Collect valid normal time windows starting point
                    normal_indices = np.where(label_array == 0)[0]
                    normal_indices = normal_indices[normal_indices < label_array.shape[0] - window_size]
                    normal_indices = np.concatenate((
                        normal_indices[normal_indices < labeled_start - window_size],
                        normal_indices[normal_indices >= labeled_end]
                    ))

                    # 3. Perform sampling and retrieve time windows
                    n_anomaly_samples = int((self.n_shots + self.n_queries) * anomaly_shot_prop)
                    n_normal_samples = (self.n_shots + self.n_queries) - n_anomaly_samples
                    normal_start_indices = np.random.choice(normal_indices, size=n_normal_samples, replace=True)
                    normal_start_indices = np.sort(normal_start_indices)
                    anomaly_start_indices = np.random.choice(labeled_indices, size=n_anomaly_samples, replace=True)
                    anomaly_start_indices = np.sort(anomaly_start_indices)
                    x_normal = [data_array[start:start + window_size] for start in normal_start_indices]
                    y_normal = [0] * n_normal_samples
                    x_anomaly = [data_array[start:start + window_size] for start in anomaly_start_indices]
                    y_anomaly = [1] * n_anomaly_samples

                    x_full = x_normal + x_anomaly
                    y_full = y_normal + y_anomaly
                else:
                    # Sampling for normal time series
                    support_start_indices = np.random.choice(label_array.shape[0]-window_size, size=(self.n_shots + self.n_queries), replace=True)
                    x_full = [data_array[start:start + window_size] for start in support_start_indices]
                    y_full = [0] * (self.n_shots + self.n_queries)

                x_spt = x_full[:self.n_shots]; x_qry = x_full[self.n_shots:]
                y_spt = y_full[:self.n_shots]; y_qry = y_full[self.n_shots:]
                x_spt = torch.tensor(x_spt, dtype=torch.float32).view(self.n_shots, window_size, 1)
                y_spt = torch.tensor(y_spt, dtype=torch.float32).view(self.n_shots)
                x_qry = torch.tensor(x_qry, dtype=torch.float32).view(self.n_queries, window_size, 1)
                y_qry = torch.tensor(y_qry, dtype=torch.float32).view(self.n_queries)
                
                self.tasks.append((x_spt, y_spt, x_qry, y_qry, task_idx))
            
        elif self.mode == 'test':
            for signal_idx, signal_value in self.signal_value_dict.items():
                label_array = self.window_label_dict[signal_idx]
                x_full = [signal_value[start:start + window_size] for start in range(label_array.shape[0]-window_size)]
                x_full = torch.tensor(x_full, dtype=torch.float32).view(-1, window_size, 1)
                y_full = [-1] * x_full.size(0) # No labels for the testing data
                y_full = torch.tensor(y_full, dtype=torch.float32).view(-1)
                self.tasks.append((x_full, y_full, signal_idx))

        elif self.mode == 'adapt':
            for signal_idx, signal_value in self.signal_value_dict.items():
                label_array = self.window_label_dict[signal_idx]
                x_full = [signal_value[start:start + window_size] for start in range(label_array.shape[0] - window_size)]
                x_full = torch.tensor(x_full, dtype=torch.float32).view(-1, window_size, 1)
                y_full = torch.tensor(label_array[window_size:], dtype=torch.float32).view(-1)

                # Shuffle the indices to randomly select shots and queries
                indices = torch.randperm(x_full.size(0))
                shots_indices = indices[:self.n_shots]
                queries_indices = indices[self.n_shots:self.n_shots + self.n_queries]

                # Sample shots and queries
                x_spt = x_full[shots_indices]
                y_spt = y_full[shots_indices]
                x_qry = x_full[queries_indices]
                y_qry = y_full[queries_indices]

                self.tasks.append((x_spt, y_spt, x_qry, y_qry, signal_idx))


        if load_from_pretrain:
            # Load the foundation model if embedding is to be used
            self.model = GTT.from_tsfoundation(signals=None, foundation_path=foundation_path, cp=None)

            # Load the weights of the head layer in the foudnation model
            mu_head_weights = self.model.estimator.get_layer('mu_head').get_weights()
            # Convert the Keras weights and biases to PyTorch tensors
            self.GTT_embedding_dim = mu_head_weights[0].shape[0] # GTT embedding dimension
            self.GTT_output_dim = mu_head_weights[0].shape[1]  # GTT output dimension
            self.GTT_head_weights = torch.tensor(mu_head_weights[0].T, dtype=torch.float32)  # Transpose to match PyTorch's shape (out_features, in_features)

            # Update the input as foundation model embeddings
            self.update_tasks_with_embeddings()
            del self.model

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    def sample_signal_indices(self):
        # If there are labeled time series
        if any(self.signal_label_dict.values()):
            signal_indices_by_label = defaultdict(list)
            for signal_idx, label in self.signal_label_dict.items():
                signal_indices_by_label[label].append(signal_idx)
            
            label_0_indices = signal_indices_by_label[0]
            label_1_indices = signal_indices_by_label[1]

            sampled_indices = [
                random.choice(label_1_indices) if random.random() < self.anomaly_sample_prop else random.choice(label_0_indices)
                for _ in range(self.n_total_tasks)
            ]
        # If there are only unlabeled normal time series
        else:
            all_signal_indices = list(self.signal_label_dict.keys())
            sampled_indices = random.choices(all_signal_indices, k=self.n_total_tasks)
        
        return sampled_indices
    
    def update_tasks_with_embeddings(self):
        """
        Update each task by replacing x_full with x_embedding, where x_embedding is the output of the GTT model.
        """
        updated_tasks = []
        total_tasks = len(self.tasks)
        logging.info(f"Starting embedding update for {total_tasks} tasks.")

        if self.mode == 'train' or self.mode == 'adapt':
            for i, task in enumerate(self.tasks):
                x_spt, y_spt, x_qry, y_qry, signal_idx = task
      
                # Generate embeddings for the combined tensor
                x_combined = torch.cat((x_spt, x_qry), dim=0)
                x_combined_embedding = self.model.get_embeddings(x_combined, window_size=x_combined.shape[1], batch_size=512)
                # Split the embeddings back into support and query sets
                x_spt_embedding = x_combined_embedding[:self.n_shots]
                x_qry_embedding = x_combined_embedding[self.n_shots:]
           
                # Update the task tuple with x_embedding instead of x
                updated_task = (x_spt_embedding, y_spt, x_qry_embedding, y_qry, signal_idx)
                updated_tasks.append(updated_task)
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                    logging.info(f"Training mode: updated embeddings for {i + 1}/{total_tasks} tasks.")

        elif self.mode == 'test':
            for i, task in enumerate(self.tasks):
                x_full, y_full, signal_idx = task
                
                # Generate embeddings using the revised get_embeddings method
                x_full_embedding = self.model.get_embeddings(x_full, window_size=x_full.shape[1], batch_size=512)
                
                # Update the task tuple with x_full_embedding instead of x_full
                updated_task = (x_full_embedding, y_full, signal_idx)
                updated_tasks.append(updated_task)
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                    logging.info(f"Testing mode: updated embeddings for {i + 1}/{total_tasks} tasks.")

        # Replace the original tasks with the updated ones
        self.tasks = updated_tasks
        logging.info("Completed embedding update for all tasks.")