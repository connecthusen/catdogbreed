## 🐾 Cat & Dog Breed Classifier

A web-based Deep Learning Image Classifier built with FastAI, Gradio, and Hugging Face Hub.
It can classify uploaded images of cats and dogs into their specific breeds using a pre-trained model (model.pkl).

---

## 🚀 Features

🔹 Built using FastAI (fastai.vision.all)

🔹 Hosted & versioned with Hugging Face Hub

🔹 Interactive Gradio UI for inference

🔹 Supports direct image uploads or example images

🔹 Lightweight, local & cloud deployable

---

## 🧠 Model Details

Base framework: FastAI

Model type: CNN (Transfer Learning)

Training dataset: Custom Cat & Dog Breed dataset

Exported file: model.pkl

Hugging Face repo: Kutti-AI/catdogbreed

---

## 🧩 Project Structure
```text
catdogbreed/
│
├── app.py               # Gradio app file
├── dog.jpg              #image file 
├── setter.jpg           #image file
├── requirements.txt     # Python dependencies
├── README.md            # This file
```

## ⚙️ Installation

# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/catdogbreed.git
cd catdogbreed

# 2️⃣ Install dependencies
pip install -r requirements.txt


or manually:

pip install fastai gradio huggingface_hub pillow

# 🔑 Hugging Face Authentication

To download the model automatically from the Hugging Face Hub:

Option 1: Set environment variable
export hf_token="your_huggingface_token"

Option 2: Set inside notebook
import os
os.environ["hf_token"] = "your_huggingface_token"

Option 3: Add manually to .env (optional)
hf_token=your_huggingface_token

#🧰 Run the App

Run the Gradio interface locally:

python app.py


Gradio will launch a local server:

Running on local URL:  http://127.0.0.1:7860

---

## 🖼️ Example Usage

You can try with:

sample_images/dog.jpg

sample_images/setter.jpg

Once launched, upload an image or select an example —
the model will output breed probabilities like:

{
  "Labrador Retriever": 0.85,
  "Golden Retriever": 0.10,
  "German Shepherd": 0.05
}

---

## 🧾 Code Explanation

# ✅ Load Model from Hugging Face
from fastai.vision.all import load_learner
from huggingface_hub import login, hf_hub_download  

login(token=os.environ["hf_token"])
model_path = hf_hub_download(repo_id="Kutti-AI/catdogbreed", filename="model.pkl")
learn = load_learner(model_path)

# ✅ Define Prediction Function
def classify_image(im):
    learn.model.eval()
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    with learn.no_bar():
        pred, idx, probs = learn.predict(im)
    return dict(zip(learn.dls.vocab, map(float, probs)))

# ✅ Gradio Interface
intf = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    examples=[['dog.jpg'], ['setter.jpg']]
)
intf.launch()

---

## 🧾 requirements.txt

```text
fastai==2.7.13
fastcore==1.5.55
torch==2.1.2
torchvision==0.16.2
transformers==4.40.2
datasets==2.13.1
numpy==1.24.4
pandas==2.2.3
matplotlib==3.7.2
spacy==3.8.7
gradio==4.44.1
timm==1.0.21
```

---

## 🌐 Deploy Options
# 🧩 On Hugging Face Spaces

Add these two files:

app.py

requirements.txt

Then push to your Hugging Face repo
→ your app will automatically deploy as an interactive Space.


# ☁️ On Localhost / Colab / Cloud

Just run python app.py — Gradio will handle local serving.

---

## 🧿 License

This project is licensed under the MIT License — free to use, modify, and share.

## ✍️ Author

👤 Husen (Kutti-AI)
💌 Machine Learning | Deep Learning 
🌐 Hugging Face Profile

“Train your model like you raise your dreams — with patience, precision, and passion.”
— Husen (Kutti-AI) 🧠💫

---
