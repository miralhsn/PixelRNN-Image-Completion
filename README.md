# 🧠 PixelRNN Image Completion with Streamlit Interface

---

## 📌 Project Overview

This project implements a **PixelRNN-based deep learning model** for image completion.

The model is trained to reconstruct missing or occluded parts of images.  
Given an occluded input image, the model predicts the missing pixels and generates a reconstructed output that approximates the original image.

In addition to the deep learning model, an interactive **Streamlit-based user interface** is developed to allow real-time image upload and reconstruction.

---

## 🎯 Objectives

- Implement PixelRNN for image completion
- Train the model using occluded images as input and original images as ground truth
- Experiment with techniques to improve reconstruction quality
- Evaluate performance visually and qualitatively
- Build an interactive Streamlit UI for user interaction

---

## 📊 Dataset

Primary Dataset:
Bedroom Occluded Images Dataset (Kaggle)

Link:
https://www.kaggle.com/datasets/mug2971/bedroom-occluded-images

Dataset Description:
- Occluded images (input)
- Corresponding original images (ground truth)
- Used for supervised image reconstruction training

Other datasets may also be used for experimentation.

---

## 🏗️ Model Architecture

### PixelRNN

PixelRNN is an autoregressive model that models the conditional distribution of pixels in an image.

The model:
- Processes image pixels sequentially
- Predicts missing pixels conditioned on previous pixels
- Learns spatial dependencies across the image

Training Setup:
- Input: Occluded image
- Target: Original image
- Loss Function: Mean Squared Error (MSE) / Reconstruction Loss
- Optimizer: Adam
- Evaluation: Visual comparison and qualitative analysis

---

## 🔬 Techniques & Improvements

To enhance model performance, the following techniques were explored:

- Data normalization
- Data augmentation
- Regularization
- Learning rate tuning
- Batch size experimentation
- Early stopping

Further improvements may include:
- Residual connections
- Gated PixelRNN
- Hybrid CNN + PixelRNN approaches

---

## 📈 Evaluation

Performance is evaluated based on:

- Visual quality of reconstructed images
- Comparison between:
  - Occluded Input
  - Reconstructed Output
  - Original Ground Truth
- Structural similarity and reconstruction fidelity

Final assessment focuses on:
- Reconstruction accuracy
- Output sharpness
- Usability of the UI

---

## 🖥️ Streamlit User Interface

An interactive Streamlit application allows users to:

- Upload an occluded image
- View the reconstructed output
- Compare with the original image
- Experience real-time inference

To run the Streamlit app:

```
streamlit run app.py
```

---

## 📂 Project Structure

```
PixelRNN-Image-Completion/
│
├── train.py
├── model.py
├── dataset.py
├── inference.py
├── app.py
│
├── data/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- PyTorch / TensorFlow
- NumPy
- OpenCV
- Streamlit
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/yourusername/PixelRNN-Image-Completion.git
cd PixelRNN-Image-Completion
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train the model:

```
python train.py
```

4. Launch the Streamlit UI:

```
streamlit run app.py
```

---

## 🔮 Future Improvements

- Integrate GAN-based refinement
- Improve edge consistency
- Add quantitative metrics (PSNR, SSIM)
- Deploy as a web application

---

## 🙏 Acknowledgment

Dataset sourced from Kaggle:
https://www.kaggle.com/datasets/mug2971/bedroom-occluded-images
