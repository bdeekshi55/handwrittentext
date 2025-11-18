Handwritten Text Recognition Project

Overview

This project aims to develop a Handwritten Text Recognition (HTR) system that can accurately recognize and transcribe handwritten text images.

Project Structure

- data/: Dataset containing handwritten text images
- models/: Trained models and checkpoints
- src/: Source code for data preprocessing, model training, and prediction
- utils/: Utility functions for data loading, image processing, and evaluation metrics

Requirements

- Python 3.8+
- TensorFlow 2.4+
- OpenCV 4.5+
- NumPy 1.20+
- Scikit-image 0.18+

Installation


bash
pip install -r requirements.txt


Dataset

- Download the IAM Handwriting Database (or your preferred dataset)
- Update data/ directory with the dataset

Training


bash
python src/train.py --data_dir data/ --model_dir models/


Prediction


bash
python src/predict.py --image_path image.jpg --model_path models/model.h5


Evaluation


bash
python src/evaluate.py --data_dir data/ --model_dir models/


Model Architecture

- Convolutional Neural Network (CNN) + Recurrent Neural Network (RNN) + Connectionist Temporal Classification (CTC)

Results

- Accuracy: 90%+ on IAM Handwriting Database

Contributing

Contributions are welcome! Please submit a pull request with your changes.

License

MIT License

Acknowledgments

- IAM Handwriting Database
- TensorFlow
- OpenCV
