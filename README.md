# Image Colorization Using KNN

This project demonstrates a simple method of **image colorization** using the **K-Nearest Neighbors (KNN)** algorithm in Python. The goal is to predict the color values of a grayscale image based on a trained color image using pixel-wise learning.

## 🧠 Technologies Used
- Python
- OpenCV
- scikit-learn
- NumPy

## 📁 Project Structure
Image-Colorization-KNN/
├── colorization_knn.py
├── grayscale/
│ └── sample_input.jpg
├── output/
│ └── colorized_output.jpg

## ⚙️ How It Works
1. A color image is used to train a KNN classifier.
2. The image is converted to LAB color space, and the L channel is used as input.
3. For a given grayscale image, the L channel is used to predict the A and B channels.
4. The predicted LAB image is converted back to BGR format to produce a colorized image.

## 🚀 Setup & Run

1. **Install dependencies**:
```bash
pip install opencv-python scikit-learn numpy
python colorization_knn.py
