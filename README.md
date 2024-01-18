# Speech Enhancement
This repository contains the implementation of a speech enhancement model, specifically focused on noise suppression. The project leverages PyTorch and PyTorch Lightning for efficient model implementation and training.

## Features
Speech enhancement model for noise suppression.
Implementation and training using PyTorch and PyTorch Lightning.
Data processing and model training on AWS SageMaker for scalability.
Utilizes datasets from the Microsoft DNS Challenge.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:

```yaml
Python >= 3.8
AWS CLI
Sagemaker Python SDK
```


## Getting Started
1. Install dependencies:

    ```bash
    pip3 install sagemaker
    ```

2. Set up AWS credentials for SageMaker.
    ```bash
    aws configure
    ```

3. Download and preprocess the [Microsoft DNS Challenge datasets](https://github.com/microsoft/DNS-Challenge).


## Training on AWS SageMaker
To train the model on AWS SageMaker, follow these steps:

1. Run data preprocessing on AWS Sagemaker:
    ```bash
    python3 process.py
    ```

3. Run the SageMaker training script:
    ```bash
    python3 train.py
    ```