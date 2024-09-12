# Data Download Instructions

To properly run the experiments and replicate the results, you need to download the telemetry dataset. Follow the steps below to prepare the dataset.

## Step 1: Download the Dataset

Download the dataset from the following S3 link:

(https://s3-us-west-2.amazonaws.com/telemanom/data.zip)

## Step 2: Extract the Dataset

After downloading the `data.zip` file, extract it. This will create a folder containing two subfolders: `train` and `test`.

## Step 3: Place Folders in the `raw` Directory

Move the `train` and `test` folders into the `raw` directory of this repository. Your directory structure should look like this:

raw/
├── train/     # Contains training data
└── test/      # Contains testing data

Once the dataset is in place, you can proceed with preprocessing, training, and evaluation as outlined in the main repository's README file.