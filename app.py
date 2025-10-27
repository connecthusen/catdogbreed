from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download
import gradio as gr
from PIL import Image

model_path = hf_hub_download(
    repo_id="Kutti-AI/dogbreed",  # <--- replace with your repo
    filename="model.pkl"
)

# Load learner
learn = load_learner(model_path)

categories=learn.dls.vocab

def classify_image(im):
    learn.model.eval()  # Force evaluation mode
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    with learn.no_bar():
        pred, idx, probs = learn.predict(im)
    return dict(zip(learn.dls.vocab, map(float, probs)))

    image=gr.Image(type="pil")
label=gr.Label()
examples=[['dog.jpg'],['setter.jpg']]

intf=gr.Interface(fn=classify_image,inputs=image,outputs=label,examples=examples)

if __name__ == "__main__":
    intf.launch()