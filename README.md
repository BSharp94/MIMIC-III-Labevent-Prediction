# MIMIC III Labevents Prediction

This project is an assignment for the AI in Healthcare graduate course at the University of Texas at Austin.

## Purpose

The purpose of this project is to see if we can predict future labevents for patients in the MIMIC III dataset.


## Project Requirements

This project requires that the use have the MIMIC III database setup locally. In order to gain access to this dataset, follow the steps at the following url:

[https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)

After gaining access to the dataset, follow the provided steps to setup the dataset in a local postgres database.

## Code

There are two steps for running the model.

### Step 1. Prepare dataset

This script runs several queries on the database in order to build the training and testing datasets. 

To run the code, run the following:

```
python prepare_dataset.py
```

### Step 2. Train model

After the first step completes, run the training script. In order to complete the training in a reasonable time, you should enable cuda support.

```
python train.py
```