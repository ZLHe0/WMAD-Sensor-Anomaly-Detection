#!/bin/bash

# Start the job
echo " "
echo "Pipeline started at $(date)"
echo " "

### Data preprocessing
python src/data_processing/preprocess.py

### For non-foundation_based models
# Training and Testing
python src/main_non_foundation_based.py 
# Evaluation
python src/evaluation/evaluate.py --method "SAD" --likelihood_threshold 0.95 --plot
python src/evaluation/evaluate.py --method "AE" --likelihood_threshold 0.95 --plot
echo "One round of mode pipeline finished at $(date)"

### For foundation_based models
# Training and Testings
python src/main_foundation_based.py --method "GTT_SAD" 
python src/main_foundation_based.py --method "GTT_SVDD" --load_data
python src/main_foundation_based.py --method "GTT_FLOS" --anomaly_sample_prop 0.5 --load_data
python src/main_foundation_based.py --method "GTT_DevNet" --anomaly_sample_prop 0.5 --load_data
# Evaluation
python src/evaluation/evaluate.py --method "GTT_SAD" --likelihood_threshold 0.85 --plot
python src/evaluation/evaluate.py --method "GTT_SVDD" --likelihood_threshold 0.85 --plot
python src/evaluation/evaluate.py --method "GTT_FLOS" --likelihood_threshold 0.85 --plot
python src/evaluation/evaluate.py --method "GTT_DevNet" --likelihood_threshold 0.85 --plot


# End of job
echo " "
echo "Non-foundation mode pipeline finished at $(date)"
echo " "
