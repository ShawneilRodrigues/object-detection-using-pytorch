# Wheat Detection using Faster R-CNN

This project demonstrates how to use a Faster R-CNN model for wheat detection in images. It utilizes the Global Wheat Detection dataset from Kaggle and PyTorch for model training and inference.

## Project Structure

- **global-wheat-detection:** Contains the dataset downloaded from Kaggle.
- **train.csv:** CSV file with bounding box annotations for training images.
- **[notebook].ipynb:** Jupyter Notebook containing the code for data preprocessing, model training, and inference.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- opendatasets
- pandas
- Pillow (PIL)
- matplotlib
- scikit-learn

## Installation

1. Install the required libraries:
2. 2. Download the Global Wheat Detection dataset from Kaggle and extract it to the project directory.

## Usage

1. Open the Jupyter Notebook ([notebook].ipynb).
2. Run the cells to preprocess the data, train the Faster R-CNN model, and perform inference on test images.
3. The notebook displays the test images with predicted bounding boxes overlaid.

## Model Training

- The Faster R-CNN model is initialized with a ResNet-50 backbone pre-trained on ImageNet.
- The model is fine-tuned on the Global Wheat Detection dataset using the Stochastic Gradient Descent (SGD) optimizer.
- The training process involves iterating over the training data, calculating the loss, and updating the model parameters.

## Inference

- The trained model is used to predict bounding boxes for wheat heads in test images.
- The predicted boxes are visualized on the test images using matplotlib.

## Acknowledgments

- The Global Wheat Detection dataset is provided by Kaggle.
- The Faster R-CNN implementation is based on the torchvision library.

## License

This project is licensed under the MIT License.
