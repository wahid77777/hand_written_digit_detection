# WEB APP (Streamlit)
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generator model (same as training)
class DigitGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(DigitGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(input)
        return img.view(img.size(0), 1, 28, 28)

# Load model
latent_dim = 100
num_classes = 10
model = DigitGenerator(latent_dim, num_classes)
model.load_state_dict(torch.load("mnist_generator.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    noise = torch.randn(5, latent_dim)
    labels = torch.full((5,), int(digit), dtype=torch.long)
    with torch.no_grad():
        images = model(noise, labels).squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
