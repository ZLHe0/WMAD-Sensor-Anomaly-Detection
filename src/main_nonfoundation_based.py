import os
import pdb
from networks.CNN import CNN_Autoencoder, CNN_encoder
from data_processing.telemetry import Telemetry_Dataset
# from trainers.AE_trainer import AETrainer
from trainers.SAD_trainer import SAD_Trainer
from trainers.AE_trainer import AE_Trainer
import pickle
import torch
import argparse
import pickle



def main(args):
    # Access the arguments
    window_size = args.window_size
    input_size = args.input_size
    hidden_size = args.hidden_size
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
    
    ### Config files
    folder_path = os.path.dirname(os.path.abspath(__file__))
    result_saving_path = os.path.join(folder_path, '../results/SAD')
    ae_result_saving_path = os.path.join(folder_path, '../results/AE')
    training_data_path = os.path.join(folder_path, '../data/processed/labeled_train_data.csv')
    testing_data_path = os.path.join(folder_path, '../data/processed/labeled_test_data.csv')

    #### Model Training - AE
    telemetry_dataset_train = Telemetry_Dataset(data_path=training_data_path, mode='train', window_size=window_size, n_shots=n_shots, n_queries=n_queries,
                                                n_total_tasks=n_total_tasks, anomaly_shot_prop=anomaly_shot_prop, anomaly_sample_prop=anomaly_sample_prop)
    ae_init_net = CNN_Autoencoder(window_size=window_size, input_size=input_size, hidden_size=hidden_size) # Initialize the network
    ae_trainer = AE_Trainer(n_epochs=n_epochs, batch_size=batch_size, device='cpu', lr=learning_rate, weight_decay=weight_decay, adapt_step=adapt_step,
                            adapt_lr=adapt_learning_rate) # Setup the trainer
    ae_trained_net = ae_trainer.train(dataset=telemetry_dataset_train, net=ae_init_net)

    #### Model Training - SAD
    init_net = CNN_encoder(window_size=window_size, input_size=input_size, hidden_size=hidden_size) # Initialize the network
    trainer = SAD_Trainer(c = None, n_epochs=n_epochs, batch_size=batch_size, device='cpu', lr=learning_rate, weight_decay=weight_decay, adapt_step=adapt_step,
                        adapt_lr=adapt_learning_rate) # Setup the trainer
    trainer.init_network_weights_from_pretraining(init_net, ae_trained_net) 
    trained_net = trainer.train(dataset=telemetry_dataset_train, net=init_net)


    # Save the dataset
    # Save the instance of trainer to a file
    # AE Model
    ae_network_saving_path = os.path.join(ae_result_saving_path, "networks")
    os.makedirs(ae_network_saving_path, exist_ok=True)
    ae_trainer_path = os.path.join(ae_network_saving_path, 'ae_trainer.pkl')
    ae_net_path = os.path.join(ae_network_saving_path, 'ae_net.pth')
    with open(ae_trainer_path, 'wb') as file:
        pickle.dump(ae_trainer, file)
    # Save the neural net to a file
    torch.save(ae_trained_net, ae_net_path)
    print(f"AE Training completed and the model is saved")

    # SAD Model
    network_saving_path = os.path.join(result_saving_path, "networks")
    os.makedirs(network_saving_path, exist_ok=True)
    trainer_path = os.path.join(network_saving_path, 'trainer.pkl')
    net_path = os.path.join(network_saving_path, 'net.pth')
    with open(trainer_path, 'wb') as file:
        pickle.dump(trainer, file)
    # Save the neural net to a file
    torch.save(trained_net, net_path)
    print(f"Training completed and the model is saved")


    ### Model Testing with MAML Adaptation
    # AE
    # Load the trainer
    with open(ae_trainer_path, 'rb') as file:
        ae_trainer = pickle.load(file)
    # Load the trained neural network
    ae_trained_net = torch.load(ae_net_path)

    # Load the dataset for fine-tuning and testing
    telemetry_dataset_finetune = Telemetry_Dataset(data_path=training_data_path, mode='adapt', window_size=window_size)
    telemetry_dataset_test = Telemetry_Dataset(data_path=testing_data_path, mode='test', window_size=window_size)
    telemetry_dataset_train_test_mode = Telemetry_Dataset(data_path=training_data_path, mode='test', window_size=window_size)

    # Perform adaptation and testing on the test set
    ae_score_saving_path = os.path.join(ae_result_saving_path, "detection")
    os.makedirs(ae_score_saving_path, exist_ok=True)
    ae_trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_test, ae_trained_net, ae_score_saving_path, save_rep=False)

    # Perform adaptation and testing on the training set (to get scores for the prior distribution)
    ae_score_on_training_set_saving_path = os.path.join(ae_result_saving_path, "detection_train")
    os.makedirs(ae_score_on_training_set_saving_path, exist_ok=True)
    ae_trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_train_test_mode, ae_trained_net, ae_score_on_training_set_saving_path, save_rep=False)



    # SAD
    # Load the trainer
    with open(trainer_path, 'rb') as file:
        trainer = pickle.load(file)
    # Load the trained neural network
    trained_net = torch.load(net_path)

    # Perform adaptation and testing on the test set
    score_saving_path = os.path.join(result_saving_path, "detection")
    os.makedirs(score_saving_path, exist_ok=True)
    trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_test, trained_net, score_saving_path, save_rep=False)

    # Perform adaptation and testing on the training set (to get scores for the prior distribution)
    score_on_training_set_saving_path = os.path.join(result_saving_path, "detection_train")
    os.makedirs(score_on_training_set_saving_path, exist_ok=True)
    trainer.adapt_and_test(telemetry_dataset_finetune, telemetry_dataset_train_test_mode, trained_net, score_on_training_set_saving_path, save_rep=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Training Script")

    # Define all the arguments
    parser.add_argument('--window_size', type=int, default=64 * 2, help='Size of the time window.')
    parser.add_argument('--input_size', type=int, default=1, help='Feature dimension (input size).')
    parser.add_argument('--hidden_size', type=int, default=8, help='CNN filter size (hidden size).')
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
    parser.add_argument('--anomaly_sample_prop', type=float, default=0.2, help='Proportion of samples that are anomalies.')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function
    main(args)