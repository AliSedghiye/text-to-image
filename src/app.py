import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

auth_token = '<your_huggingface_auth_token>'

modelid = "stabilityai/sd-turbo"
device = "cpu" ## Change to "cuda" if you have a GPU

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        modelid,
        use_auth_token=auth_token
    )
    pipe.to(device)
    return pipe

pipe = load_pipeline()

st.title("Stable Bud - Streamlit Edition (Turbo)")
prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        image = pipe(prompt, guidance_scale=0.0).images[0]
        image.save('generatedimage.png')
        st.image(image, caption="Generated Image", use_column_width=True)