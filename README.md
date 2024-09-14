[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A2tcAnZG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15921635&assignment_repo_type=AssignmentRepo)
# emlov3-session-02

# PyTorch Docker Assignment

Welcome to the PyTorch Docker Assignment. This assignment is designed to help understand and work with Docker and PyTorch. 

## Assignment Overview

This project trains a neural network on the MNIST dataset using PyTorch. The project is containerized with Docker, making it easy to reproduce the environment. In this assignment contains:

1. Create a Dockerfile for a PyTorch (CPU version) environment.
2. Keep the size of your Docker image under 1GB (uncompressed).
3. Train any model on the MNIST dataset inside the Docker container.
4. Save the trained model checkpoint to the host operating system.
5. Add an option to resume model training from a checkpoint.

## Starter Code

The provided starter code in train.py provides a basic structure for loading data, defining a model, and running training and testing loops. And with this submission, the code is completed.

## How to Run the Code Using Docker
Below are the instructions to build and run the code using Docker.

### Requirements
- Docker installed on your machine.

#### Dockerfile Overview
The provided `Dockerfile` does the following:

1. **Base Image:** Uses `python:3.9-slim` as the base image.
2. **Working Directory:** Sets `/workspace` as the working directory inside the container.
3. **Package Installation:** Installs specific versions of `numpy`, `torch`, and `torchvision` using `pip`.
4. **Copy Files:** Copies train.py to the working directory.
5. **Command to Execute:** The default command to run the training script is python `train.py`.

#### How to Build and Run the Docker Container
##### Step 1: Build the Docker Image
Navigate to the directory containing the `Dockerfile` and run the following command to build the Docker image:


```
docker build -t mnist-trainer:latest .

```
This command:

- Builds the Docker image and tags it as `mnist-trainer:latest`.
  
##### Step 2: Run the Docker Container
Once the image is built, you can run the container using the following command:


```
docker run --rm -it -v $(pwd)/data:/workspace/data mnist-trainer:latest

```
Explanation:

- `--rm`: Automatically removes the container once it exits.
- `-it`: Runs the container interactively, allowing you to see the training output in real time.
- `-v $(pwd)/data:/workspace/data`: Mounts the `data` directory from your host system into the container at `/workspace/data`, allowing MNIST data and model checkpoints to persist between runs.
- `mnist-trainer:latest`: Specifies the Docker image to run.

##### Step 3: Running with Checkpoint Resume
To resume training from a checkpoint, first make sure a model checkpoint exists at `./model_checkpoint.pth`. Then, add the `--resume` flag when running the container:


```
docker run --rm -it -v $(pwd)/data:/workspace/data mnist-trainer:latest --resume

```
This will load the existing checkpoint and continue training.

##### Additional Docker Commands
- **To view the logs:** Use the following command to check the logs of the running container:


```
docker logs <container-id>

```
- **To save the model:** After training, the model checkpoint will be saved in `./model_checkpoint.pth` on your local machine.

##### Notes
- The model architecture and training script can be modified in `train.py`.
- The container will automatically download the MNIST dataset during the training process if not already present.

## Test Results

All the tests run with the script `tests/grading.sh` completed successfully on gitpod.

## Submission

After the assignment completion, push code to the Github repository. The Github Actions workflow will automatically build the Docker image, run  training script, and check if the assignment requirements have been met. Check the Github Actions tab for the results of these checks. It is made sure that all checks are passing before the assignment submission.
