import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the same VAE model used in training
class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 400)
        self.fc21 = torch.nn.Linear(400, 20)  # Mean
        self.fc22 = torch.nn.Linear(400, 20)  # Log variance
        self.fc3 = torch.nn.Linear(20, 400)
        self.fc4 = torch.nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

# Load the trained model
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location='cpu'))
model.eval()

# Streamlit UI
st.title("🧠 MNIST Handwritten Digit Generator")

# Digit selector (this won't affect generation unless you add class-conditional logic)
digit = st.selectbox("Select a digit (0–9):", list(range(10)))

if st.button("Generate 5 Samples"):
    st.write(f"Generating 5 samples of digit {digit} (random VAE samples)")

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))

    for i in range(5):
        z = torch.randn(1, 20)  # Sample from latent space
        sample = model.decode(z).detach().numpy().reshape(28, 28)
        axs[i].imshow(sample, cmap="gray")
        axs[i].axis("off")

    st.pyplot(fig)
