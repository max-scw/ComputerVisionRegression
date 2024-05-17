# Computer Vision Regression
Toy example for a Computer Vision CNN with a regression head

## Overview

This project is a [PyTorch](https://pytorch.org/)-based implementation of a regression head added to an image recognition neural network, specifically [MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html) (see [model.py](model.py)). 
The purpose of this project is to demonstrate how to adapt an image classification model for regression tasks. The project includes a function to generate toy data for testing and experimentation purposes: [create_toy_data.py](create_toy_data.py).


## Project Structure
````
ComputerVisionRegression
+-- utils  # Contains utility functions for data processing and other helper functions.
  |-- dataset.py  # creates a dataset generator
|-- create_toy_data.py  # run to create a folder "datasets > toydata" with toy images and split the data into training/validation/testing sets
|-- model.py  # creates the model architecture
|-- README.md
|-- requirements.txt
|-- train.py
````


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author
 - max-scw

## Status
active (low priority)