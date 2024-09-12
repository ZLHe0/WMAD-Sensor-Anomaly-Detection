import os
import sys
import pdb
from data_processing.telemetry import Telemetry_Dataset
import importlib
import pickle
import torch
import argparse

def main(args):
    # Access the arguments
    method = args.method
    load_data = args.load_data
    window_size = args.window_size
    learning_rate = args.learning_rate
    adapt_learning_rate = args.adapt_learning_rate
    adapt_step = args.adapt_step
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    n_shots = args.n_shots
    n_queries = args.n_queries
    n_total_tasks = args.n_total_tasks
    anomaly_shot_prop = args.anomaly_shot_prop
    anomaly_sample_prop = args.anomaly_sample_prop
    
    ### Path
    folder_path = os.path.dirname(os.path.abspath(__file__))
    result_saving_path = os.path.join(folder_path, f'../results/{method}')
    training_data_path = os.path.join(folder_path, '../data/processed/labeled_train_data.csv')
    testing_data_path = os.path.join(folder_path, '../data/processed/labeled_test_data.csv')
    save_path_train = os.path.join(folder_path, '../data/processed/telemetry_embedding_train.pkl')
    save_path_test = os.path.join(folder_path, '../data/processed/telemetry_embedding_test.pkl')
    save_path_finetune = os.path.join(folder_path, '../data/processed/telemetry_embedding_finetune.pkl')
    save_path_test_on_training_set = os.path.join(folder_path, '../data/processed/telemetry_embedding_test_on_training_set.pkl')
    foundation_path = os.path.join(folder_path, 'networks/GTT_main/checkpoints/GTT-small')

    # Data loading
    if(load_data == False):
        # Create the dataset
        telemetry_dataset_train = Telemetry_Dataset(data_path=training_data_path, mode='train', window_size=window_size, n_shots=n_shots, n_queries=n_queries,
                                                    n_total_tasks=n_total_tasks, anomaly_shot_prop=anomaly_shot_prop, anomaly_sample_prop=anomaly_sample_prop,
                                                    load_from_pretrain=True, foundation_path=foundation_path)
        telemetry_dataset_test = Telemetry_Dataset(data_path=testing_data_path, mode='test', window_size=window_size, load_from_pretrain=True, foundation_path=foundation_path)
        telemetry_dataset_finetune = Telemetry_Dataset(data_path=training_data_path, mode='adapt', window_size=window_size, load_from_pretrain=True, foundation_path=foundation_path)
        telemetry_dataset_test_on_training_set = Telemetry_Dataset(data_path=training_data_path, mode='test', window_size=window_size, load_from_pretrain=True, foundation_path=foundation_path)
        # Save dataset
        os.makedirs(os.path.dirname(save_path_train), exist_ok=True)
        with open(save_path_train, 'wb') as f:
            pickle.dump(telemetry_dataset_train, f)
        with open(save_path_test, 'wb') as f:
            pickle.dump(telemetry_dataset_test, f)
        with open(save_path_finetune, 'wb') as f:
            pickle.dump(telemetry_dataset_finetune, f)
        with open(save_path_test_on_training_set, 'wb') as f:
            pickle.dump(telemetry_dataset_test_on_training_set, f)
        print("Dataloader created and saved")
    else:
        with open(save_path_train, 'rb') as f:
            telemetry_dataset_train = pickle.load(f)
        with open(save_path_test, 'rb') as f:
            telemetry_dataset_test = pickle.load(f)
        with open(save_path_finetune, 'rb') as f:
            telemetry_dataset_finetune = pickle.load(f)
        with open(save_path_test_on_training_set, 'rb') as f:
            telemetry_dataset_test_on_training_set = pickle.load(f)
        print("Dataloader loaded")

    # Dynamically import the correct network and trainer based on the method
    if method in ["GTT_SAD", "GTT_SVDD"]:
        network_module = importlib.import_module('networks.GTT_FFN')
        trainer_module = importlib.import_module('trainers.SAD_trainer')
        NetworkClass = getattr(network_module, 'GTT_FFN')
        TrainerClass = getattr(trainer_module, 'SAD_Trainer')

        # If method is GTT_SVDD, set all windows as normal
        if method == "GTT_SVDD":
            for i, task in enumerate(telemetry_dataset_train.tasks):
                x_spt, y_spt, x_qry, y_qry, task_idx = task
                y_spt = torch.zeros_like(y_spt)
                y_qry = torch.zeros_like(y_qry)
                telemetry_dataset_train.tasks[i] = (x_spt, y_spt, x_qry, y_qry, task_idx)

        # Initialize the network
        init_net = NetworkClass(
            window_size=window_size, 
            embedding_size=telemetry_dataset_train.GTT_embedding_dim,
            rep_dim=telemetry_dataset_train.GTT_output_dim, 
            pre_trained_weights=telemetry_dataset_train.GTT_head_weights.clone().detach()
        )

        # Setup the trainer
        trainer = TrainerClass(
            c=None, n_epochs=n_epochs, batch_size=batch_size, device='cpu', 
            lr=learning_rate, weight_decay=weight_decay, adapt_step=adapt_step, 
            adapt_lr=adapt_learning_rate
        )

    elif method == "GTT_FLOS":
        network_module = importlib.import_module('networks.GTT_FLOS')
        trainer_module = importlib.import_module('trainers.FLOS_trainer')
        NetworkClass = getattr(network_module, 'GTT_FLOS')
        TrainerClass = getattr(trainer_module, 'FLOS_Trainer')

        # Initialize the network
        init_net = NetworkClass(embedding_size=telemetry_dataset_train.GTT_embedding_dim)

        # Setup the trainer
        trainer = TrainerClass(
            n_epochs=n_epochs, batch_size=batch_size, device='cpu', 
            lr=learning_rate, weight_decay=weight_decay, 
            adapt_step=adapt_step, adapt_lr=adapt_learning_rate
        )

    elif method == "GTT_DevNet":
        network_module = importlib.import_module('networks.GTT_DevNet')
        trainer_module = importlib.import_module('trainers.DevNet_trainer')
        NetworkClass = getattr(network_module, 'GTT_DevNet')
        TrainerClass = getattr(trainer_module, 'DevNet_Trainer')

        # Initialize the network
        init_net = NetworkClass(
            window_size=window_size, 
            embedding_size=telemetry_dataset_train.GTT_embedding_dim,
            rep_dim=telemetry_dataset_train.GTT_output_dim, 
            pre_trained_weights=telemetry_dataset_train.GTT_head_weights.clone().detach()
        )

        # Setup the trainer
        trainer = TrainerClass(
            n_epochs=n_epochs, batch_size=batch_size, device='cpu', 
            lr=learning_rate, weight_decay=weight_decay, 
            adapt_step=adapt_step, adapt_lr=adapt_learning_rate
        )

    print(f"{method} training started")
    trained_net = trainer.train(dataset=telemetry_dataset_train, net=init_net)
    # Save the instance of trainer to a file
    os.makedirs(result_saving_path, exist_ok=True)
    network_saving_path = os.path.join(result_saving_path, "networks")
    os.makedirs(network_saving_path, exist_ok=True)
    trainer_path = os.path.join(network_saving_path, 'trainer.pkl')
    net_path = os.path.join(network_saving_path, 'net.pth')
    with open(trainer_path, 'wb') as file:
        pickle.dump(trainer, file)
    # Save the neural net to a file
    torch.save(trained_net, net_path)
    print(f"{method} training completed and the model is saved")

    ### Model Testing
    # Load the trainer
    with open(trainer_path, 'rb') as file:
        trainer = pickle.load(file)
    # Load the trained neural network
    trained_net = torch.load(net_path)

    # Perform testing and save the results
    score_saving_path = os.path.join(result_saving_path, "detection")
    os.makedirs(score_saving_path, exist_ok=True)
    trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_test, trained_net, score_saving_path, save_rep = False)

    # Perform testing and save the results
    score_training_set_saving_path = os.path.join(result_saving_path, "detection_train")
    os.makedirs(score_training_set_saving_path, exist_ok=True)
    trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_test_on_training_set, trained_net, score_training_set_saving_path, save_rep = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Script")

    # Define all the arguments
    parser.add_argument('--method', type=str, required=True, choices=['GTT_SAD', 'GTT_SVDD', 'GTT_FLOS', 'GTT_DevNet'], 
                        help='Specify the method for the pipeline.')
    parser.add_argument('--load_data', action='store_true', help='Load the saved dataloader. If not provided, a new dataloader will be created.')    
    parser.add_argument('--window_size', type=int, default=64 * 2, help='Size of the time window.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--adapt_learning_rate', type=float, default=0.01, help='Learning rate for adaptation.')
    parser.add_argument('--adapt_step', type=int, default=1, help='Number of steps in adaptation phase.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for the optimizer.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--n_shots', type=int, default=20, help='Number of shots (training examples) per task.')
    parser.add_argument('--n_queries', type=int, default=30, help='Number of query examples per task.')
    parser.add_argument('--n_total_tasks', type=int, default=1000, help='Total number of tasks in training.')
    parser.add_argument('--anomaly_shot_prop', type=float, default=0.1, help='Proportion of shots that are anomalies.')
    parser.add_argument('--anomaly_sample_prop', type=float, default=0.2, help='Proportion of windows that are anomalies.')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function
    main(args)