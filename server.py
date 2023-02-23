import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
from models.WGAN_Generator import Generator
from models.DCGANGenerator_0 import Generator as dcgen
from models.DCGANGenerator_1 import Generator as gen 

st.header("Face Generation")
st.subheader("")

to_image = transforms.ToPILImage() #TODO put to the main function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_image_full_model(model_class, path):   
    model = model_class
    model = torch.load(path)
    model.eval()
    noise =  torch.randn(1, 100, 1, 1)#, device=device)
    img = model(noise).to(device)
    o_path = os.path.join('data','output','1.png')
    save_image(img, o_path, normalize=True)  
    img = torch.squeeze(img)
    img = to_image(img)
    
    return img, o_path

if st.button('Generate face with DCGAN'):
    model_class =  dcgen()
    path = os.path.join("data", "models", "dcgan_model_0.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width=256 , output_format='PNG')

if st.button('Generate face with DCGAN 2'):
    model_class =  gen()
    path = os.path.join("data", "models", "dcgan_model_1.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')

if st.button('Generate face with WGAN verion 1'):
    model_class = Generator()
    print(model_class)
    path = os.path.join("data", "models", "wgan_version_0.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')

if st.button('Generate face with WGAN verion 2'):
    model_class = Generator()
    print(model_class)
    path = os.path.join("data", "models", "wgen_version_1.pt") 
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')