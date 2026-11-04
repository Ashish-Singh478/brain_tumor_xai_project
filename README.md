# 🧠 Brain Tumor Classification with Explainable AI (Grad-CAM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)

## The Problem
Deep learning models often achieve high accuracy in medical image analysis, but their "black-box" nature makes clinical adoption difficult. A doctor cannot trust a model's prediction if the model cannot explain *why* it made that decision.

## Our Solution
This project implements an end-to-end Deep Learning pipeline to classify MRI brain scans into four categories (Glioma, Meningioma, Pituitary, and No Tumor) using a **ResNet50** architecture. It achieved an accuracy of **>98%**.

More importantly, it integrates **Explainable AI (XAI)** via **Grad-CAM**. Instead of just giving a prediction, the model generates a visual heatmap over the MRI scan, highlighting the exact neurological region it focused on to make its decision. This brings transparency and clinical trust to the AI's predictions.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision (ResNet50)
* **Explainable AI:** Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Frontend UI:** Streamlit
* **Image Processing:** OpenCV, PIL, NumPy

## 🚀 How to Run the Project Locally

Because the trained model weights (`best_model.pt`) and datasets are large, they are not hosted directly on this GitHub repository. Follow these steps to set up the project on your local machine:

**1. Clone the repository**
```bash
git clone [https://github.com/Ashish-Singh478/brain_tumor_xai_project.git](https://github.com/Ashish-Singh478/brain_tumor_xai_project.git)
cd brain_tumor_xai_project
```

**2. Set up a virtual environment**

Bash
```bash
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**

Bash
```bash
pip install -r requirements.txt
```

**4. Download the Model Weights**

- Download the pre-trained ResNet50 model (`best_model.pt`) from this link: **[Google Drive](https://drive.google.com/file/d/1qQqbv_g8VeJpD-GRfQbImE_E6Gd__Yh7/view?usp=sharing)**
- Create a folder named `artifacts` in the root directory and place the `best_model.pt` file inside it.

**5. Start the Application**

Bash
```bash
streamlit run app.py
```

*The Streamlit web dashboard will automatically open in your default browser at `localhost:8501`.*

## 📂 Project Structure

Plaintext
```graphql
brain_tumor_xai_project/
│
├── artifacts/             # (Ignored in Git) Place best_model.pt here
├── data/                  # (Ignored in Git) Training & testing MRI scans
├── src/                   # Source code for training, model architecture, and XAI
├── .gitignore             # Ignored heavy files (venv, pycache, artifacts)
├── app.py                 # Main Streamlit web application
└── requirements.txt       # Project dependencies
```

## 👥 Team & Credits

This project was developed by undergraduate students of B.Tech (Computer Science & Engineering - Artificial Intelligence) at Noida Institute of Engineering and Technology (NIET), Greater Noida:

- **Ashish Singh** (Roll No: 2401331520054)
- **Ayush Yadav** (Roll No: 2401331520061)
- **GeeShiv Raj** (Roll No: 2401331520075)

**Supervised by:** Dr.Varsha jotwani (Assistant Professor, CSE-AI)

**HOD:** Dr. Anand Kumar Gupta

---
