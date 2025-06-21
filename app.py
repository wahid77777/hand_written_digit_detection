import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator model matching your training code
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # One-hot encode labels
        labels_embed = torch.zeros(labels.size(0), self.num_classes).to(device)
        labels_embed.scatter_(1, labels.unsqueeze(1), 1)
        x = torch.cat([z, labels_embed], dim=1)
        output = self.main(x)
        return output.view(-1, 1, 28, 28)

# Load generator
latent_dim = 100
num_classes = 10
G = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)

# Load model weights
try:
    G.load_state_dict(torch.load("mnist_generator.pth", map_location=device))
    G.eval()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator (0–9)")
digit = st.selectbox("Choose a digit to generate (0–9)", list(range(10)))

if st.button("Generate 5 Images"):
    with torch.no_grad():
        z = torch.randn(5, latent_dim).to(device)
        labels = torch.full((5,), int(digit), dtype=torch.long).to(device)
        generated_images = G(z, labels).cpu().squeeze().numpy()

    # Display images
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(generated_images[i], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
