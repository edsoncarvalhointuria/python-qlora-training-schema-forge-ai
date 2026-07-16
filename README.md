# AI Training Files

This repository contains the files used to generate and train an AI model.

## Overview

The training pipeline consists of two main steps:

1. **Dataset Generation**
   - Training samples are automatically generated using an AI model through an API.
   - The generated data includes both synthetic JSON examples and their corresponding schemas.

2. **Model Training**
   - The generated dataset is published on the Hugging Face Hub.
   - A LoRA fine-tuning process is performed using the published training and test datasets.

## Purpose

This repository is intended to store the scripts and configuration used during the training process. It is **not** the trained model itself.

The resulting datasets (train/test) and the LoRA adapter are published separately on Hugging Face.