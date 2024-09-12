# Weakly-Supervised Multi-Sensor Anomaly Detection (WMAD)

## Overview

This repository contains the implementation of the **Weakly-Supervised Multi-Sensor Anomaly Detection (WMAD)** framework, designed for robust anomaly detection in industrial sensor data. The framework addresses the challenges of label sparsity and sensor heterogeneity by using importance sampling and meta-learning to learn an adaptive hypersphere to separate normal and anomalous data in different sensors with minimal labeled data.

WMAD is evaluated on large-scale proprietary industrial datasets and publicly available telemetry datasets, demonstrating state-of-the-art performance across various settings.

## Repository Structure

```
.
├── data
│   ├── raw                   # Raw input data
│   └── README.txt            # Instructions for accessing the public dataset
├── src
│   ├── data_processing       # Data preprocessing scripts
│   │   ├── preprocess.py     # Main preprocessing script
│   │   └── telemetry.py      # Dataset generation script
│   ├── evaluation            # Evaluation scripts
│   │   └── evaluate.py       # Main evaluation script
│   ├── networks              # Model architectures
│   ├── trainers              # Training, testing and adapting logic
│   ├── main_foundation_based.py     # Main pipeline for foundation-based models
│   ├── main_nonfoundation_based.py  # Main pipeline for non-foundation-based models
│   └── run_experiment.sh     # Bash file for reproducing the results
└── README.md                 # This readme file
```


## Reproducing Results

To reproduce results in the paper, execute the following command. This script performs data preprocessing, model training and testing, and evaluation in a single run:

```bash
pip install -r requirements.txt
bash run_experiment.sh
```

This will:

1. Preprocess the raw data and store it in the `processed` directory.
2. Train the WMAD model along with the baseline models.
3. Evaluate the model and store results in the `results` directory.

## Step-by-Step Usage

### Environment Setup

Install the necessary dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### 1. Data Preparation

- Place the raw data in the `data/raw` directory.
- For the public dataset, refer to the `README.txt` file located in `data/` for instructions on downloading and placing the dataset.

### 2. Data Preprocessing

To preprocess the raw data:

```bash
python src/data_processing/preprocess.py
```

This will process the raw data and store the processed data in the `processed` folder.

### 3. Model Training

You can train foundation-based models by running:

```bash
python src/foundationbase.py --method "SAD"
```

Similarly, for non-foundation-based models:

```bash
python src/nonfoundationbase.py --method "GTT_SAD"
```

### 4. Model Evaluation

After training, you can evaluate the model using:

```bash
python src/evaluation/evaluate.py --method "GTT_SAD" --threshold 0.85
```

This will output evaluation metrics and store them in the `results` folder.

## License

This project is licensed under the MIT License.