import sys
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import pdb
from trainers.base_trainer import BaseTrainer
from trainers import logger 
from trainers.MAML_trainer import MAML


# Adapted from https://github.com/lukasruff/Deep-SAD-PyTorch
class SAD_Trainer(BaseTrainer):

    def __init__(self, c, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0, eta: float = 1.0,
                 adapt_step: int = 5, adapt_lr: float = 0.01):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader,
                         adapt_step, adapt_lr)
        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None # Center of the hypersphere
        self.eps = 1e-6
        self.eta = eta # Hyperparameter that adjusts the weight of the supervised loss component

        # Results
        self.train_time = None
        self.test_time = None

    # Model initilization
    def init_network_weights_from_pretraining(self, sad_net, pr_net):
        """Initialize the Deep SVDD network weights from the encoder weights of a pretraining model."""

        net_dict = sad_net.state_dict()
        pr_net_dict = pr_net.state_dict()

        # Filter relevant network keys
        pr_net_dict = {k: v for k, v in pr_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(pr_net_dict)
        sad_net.load_state_dict(net_dict)

    def train(self, dataset, net):
        net = net.to(self.device)
        # Set data loader and optimizer
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.n_jobs_dataloader)
        
        # create the maml learner
        maml = MAML(net, lr=self.adapt_lr, first_order=False, allow_unused=True)

        # Set the optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        loss_values = []


        for epoch in range(self.n_epochs):
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader: # For each task
                meta_train_loss = 0.0
                x_spt, y_spt, x_qry, y_qry, _ = data
                x_spt, x_qry = x_spt.to(self.device), x_qry.to(self.device)
                y_spt, y_qry = y_spt.to(self.device), y_qry.to(self.device)

                tasks_per_batch = x_spt.shape[0]
                # for each task in the batch
                for i in range(tasks_per_batch):
                    learner = maml.clone()

                    # perform adaptation
                    for _ in range(self.adapt_step):
                        support_preds = learner(x_spt[i])
                        support_dist = torch.sum((support_preds - self.c) ** 2, dim = 1)
                        support_losses = torch.where(y_spt[i] == 0, support_dist, self.eta * ((support_dist + self.eps) ** (-y_spt[i].float())))
                        support_loss = torch.mean(support_losses)
                        learner.adapt(support_loss)

                    # calculate the loss
                    query_preds = learner(x_qry[i])
                    query_dist = torch.sum((query_preds - self.c) ** 2, dim=1)
                    query_losses = torch.where(y_qry[i] == 0, query_dist, self.eta * ((query_dist + self.eps) ** (-y_qry[i].float())))
                    query_loss = torch.mean(query_losses)
                    meta_train_loss += query_loss
                
                meta_train_loss = meta_train_loss / tasks_per_batch
                optimizer.zero_grad()
                meta_train_loss.backward()
                optimizer.step()

                loss_epoch += meta_train_loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            loss_values.append(loss_epoch / n_batches)
            logger.info(' Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
    
        
        phase_name = 'training'
        plt.figure(figsize=(8, 6))
        plt.semilogy(range(1, self.n_epochs + 1), loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'{phase_name} phase: training loss vs epoch')
        plt.show()               

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Finished training.')

        return maml                    

    def adapt_and_test(self, finetune_dataset, testing_dataset, maml_learner, output_data_path, save_rep=False):
        # Set the device for the network
        maml_learner = maml_learner.to(self.device)
        
        # Create data loaders for the finetune and testing datasets
        finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=1, shuffle=False,
                                                    num_workers=self.n_jobs_dataloader)
        test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                num_workers=self.n_jobs_dataloader)

        # Initialize the list for saving scores
        idx_score = []
        if save_rep: idx_rep_score = []

        # Dictionary to store adapted learners for each signal
        adapted_learners = {}

        # Adaptation Phase
        for data in finetune_loader:
            x_spt, y_spt, x_qry, y_qry, signal_idx = data
            x_spt, x_qry = x_spt.to(self.device), x_qry.to(self.device)
            y_spt, y_qry = y_spt.to(self.device), y_qry.to(self.device)

            # Create a clone of the MAML learner for adaptation
            learner = maml_learner.clone()

            signal_name = signal_idx[0] 
            logger.info(f'Adapting model for signal: {signal_name}')

            start_time = time.time()
            tasks_per_batch = x_spt.shape[0]
            meta_adapt_error = 0.0
            
            for i in range(tasks_per_batch):
                for _ in range(self.adapt_step):
                    support_preds = learner(x_spt[i])
                    support_dist = torch.sum((support_preds - self.c) ** 2, dim=1)
                    support_losses = torch.where(y_spt[i] == 0, support_dist, self.eta * ((support_dist + self.eps) ** (-y_spt[i].float())))
                    support_loss = torch.mean(support_losses)
                    learner.adapt(support_loss)

                query_preds = learner(x_qry[i])
                query_dist = torch.sum((query_preds - self.c) ** 2, dim=1)
                query_losses = torch.where(y_qry[i] == 0, query_dist, self.eta * ((query_dist + self.eps) ** (-y_qry[i].float())))
                query_loss = torch.mean(query_losses)
                meta_adapt_error += query_loss

            meta_adapt_error = meta_adapt_error / tasks_per_batch
            adapt_time = time.time() - start_time

            logger.info(f'Signal: {signal_name}, Adaptation time: {adapt_time:.3f}s, Average Loss: {meta_adapt_error:.6f}')

            # Save the adapted learner for this specific signal
            adapted_learners[signal_name] = learner

        # Testing Phase
        logger.info('Starting testing...')
        start_time = time.time()

        with torch.no_grad():
            for data in test_loader:
                x_full, y_full, signal_idx = data
                x_full = x_full.to(self.device)
                y_full = y_full.to(self.device)
                signal_name = signal_idx[0]  # signal_idx is a string with the signal name

                learner = adapted_learners[signal_name]  # Retrieve the adapted learner for this signal
                learner.eval()

                support_preds = learner(x_full[0])
                support_scores = torch.sum((support_preds - self.c) ** 2, dim=1)
                    
                if save_rep:  # Save the representations for each time window
                    outputs_np = support_preds.cpu().data.numpy()
                    outputs_list = []
                    for window_idx in range(outputs_np.shape[0]):
                        rep = support_preds[window_idx, :]
                        outputs_list.append(rep)
                    idx_rep_score += list(zip([signal_name] * len(outputs_list),
                                            list(range(len(support_scores))),
                                            outputs_list,
                                            support_scores.cpu().data.numpy().tolist()))

                idx_score += list(zip([signal_name] * len(support_scores),
                                    list(range(len(support_scores))),
                                    support_scores.cpu().data.numpy().tolist()))

        if save_rep:
            dtype = np.dtype([('signal_idx', str), ('idx', int), ('rep', object), ('scores', float)])
            structured_array = np.array(idx_rep_score, dtype=dtype)
            np.save(os.path.join(output_data_path, "idx_rep_score.npy"), structured_array)

        idx_label_score_df = pd.DataFrame(idx_score, columns=['signal_idx', 'timestamp', 'scores'])
        idx_label_score_df.to_csv(os.path.join(output_data_path, "idx_score.csv"), index=False)

        self.test_time = time.time() - start_time
        logger.info(f'Testing time: {self.test_time:.3f}s')
        logger.info('Finished testing.')


    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                x_spt, y_spt, x_qry, y_qry, _ = data
                x_spt.to(self.device)
                x_qry.to(self.device)

                tasks_per_batch = x_spt.shape[0]
                # for each task in the batch
                for i in range(tasks_per_batch):

                    # Only use the normal data points to construct the initial value
                    # x_spt: (batch_size, channel_dim, seq_length)
                    # y_spt: (batch_size, is_anomaly)
                    spt_mask = (y_spt[i] == 0)
                    x_spt_i_normal = x_spt[i][spt_mask]
                    qry_mask = (y_qry[i] == 0)
                    x_qry_i_normal = x_qry[i][qry_mask]

                    # Calculate the mapping of support and query data
                    support_preds = net(x_spt_i_normal)                    
                    query_preds = net(x_qry_i_normal)

                    n_samples += support_preds.shape[0]
                    n_samples += query_preds.shape[0]

                    c += torch.sum(support_preds, dim=0)
                    c += torch.sum(query_preds, dim=0)

            c /= n_samples
            
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
