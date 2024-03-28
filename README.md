# vae-pytorch

## Project Overview
This project implements a Variational Autoencoder (VAE) to generate and reconstruct images from the MNIST dataset. The VAE model is trained to learn a latent representation of handwritten digits, enabling it to generate new images similar to those found in the dataset. The project includes functionality for training and testing the VAE model, saving sample reconstructions and generated images, and additional utilities for annotating images and converting a series of images into a GIF.

## Installation
Before running the project, ensure that you have Python and PyTorch installed on your machine. The project also requires additional libraries such as PIL (Pillow) for image manipulation tasks.

1. Clone the repository to your local machine.
2. Install the required Python packages:
```commandline
pip install torch torchvision pillow
```

## Running the Project
To run the project, navigate to the project directory and execute the main script:

```bash
python main.py
```

## Result
![test_dataset](https://github.com/dev-jinwoohong/vae-pytorch/assets/70004933/08cd7383-9a0b-4025-a28c-cabbafd13fc5) <br>
![sample](https://github.com/dev-jinwoohong/vae-pytorch/assets/70004933/5161bfac-3518-41fb-b47e-2300dff80555)



## Directory Structure
`./data`: The MNIST dataset will be downloaded and stored in this directory. <br>
`./samples_test`: This directory contains sample images generated by the VAE model at different epochs. <br>
`./results_test`: This directory holds reconstruction comparison images for each epoch during the test phase. <br>

## Key Components
`VAE`: Defines the Variational Autoencoder model architecture, including the encoder and decoder networks. <br>
`train()`: Trains the VAE model using the MNIST training dataset. <br>
`test()`: Tests the VAE model on the MNIST test dataset and saves sample reconstruction images. <br>
`image_annotate()`: Utility function to annotate images with text labels. This is used for marking images with their respective epochs or other relevant information. <br>
`image_to_gif()`: Converts a series of images into a GIF. This can be used to visualize the progression of the model's performance over different epochs. <br>
## Utilities
`utils.py`: Contains utility functions such as `image_annotate` for adding text annotations to images and `image_to_gif` for creating a GIF from a sequence of images.
## Customization
You can customize the training process by modifying the epochs variable in the main script. Additionally, you can experiment with different architectures and hyperparameters in the VAE class to explore their impact on the model's performance.

## Contributing
Contributions to the project are welcome! Please feel free to submit pull requests with improvements or file issues for any bugs or suggestions.
