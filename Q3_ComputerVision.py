import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# ---------------------------------------------------------
# Step 1: Import Libraries & Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Webcam Classifier",
    layout="centered"
)

st.title("Question 3: Real-Time Webcam Classifier")
st.write("### Model: ResNet-18 | Input: Live Webcam")

# ---------------------------------------------------------
# Step 2: Download ImageNet Labels
# ---------------------------------------------------------
@st.cache_resource
def get_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        r = requests.get(url)
        labels = r.text.strip().split("\n")
        return labels
    except:
        st.error("Error downloading labels.")
        return []

labels = get_labels()

# ---------------------------------------------------------
# Step 3: Load Pre-trained ResNet-18 Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Load ResNet18 with default pre-trained weights
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# ---------------------------------------------------------
# Step 4: Define Image Preprocessing Pipeline
# ---------------------------------------------------------
# Must match ResNet-18 training data stats
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------
# Step 5: Capture & Process Image (Webcam)
# ---------------------------------------------------------
st.subheader("Step 5: Capture Image")
st.info("Click 'Take Photo' to activate your webcam.")

# This activates the laptop camera [cite: 287]
img_buffer = st.camera_input("Take a photo")

if img_buffer is not None:
    # Convert captured buffer to PIL Image
    image = Image.open(img_buffer).convert("RGB")
    
    # Preprocess the image to Tensor
    input_tensor = preprocess(image)
    
    # Create mini-batch (Add batch dimension) [cite: 287]
    input_batch = input_tensor.unsqueeze(0) 

    # ---------------------------------------------------------
    # Step 6: Prediction & Results
    # ---------------------------------------------------------
    st.subheader("Step 6: Classification Results")
    
    # Run Inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Apply Softmax to get probabilities [cite: 288]
    probabilities = F.softmax(output[0], dim=0)
    
    # Get Top 5 Predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Format Results into a List
    results = []
    for i in range(5):
        class_name = labels[top5_catid[i]]
        score = top5_prob[i].item()
        results.append([class_name, f"{score:.4f}"])

    # Display as Table
    df = pd.DataFrame(results, columns=["Class Label", "Probability"])
    st.table(df) # Shows results in a table [cite: 288]
    
    # Highlight Winner
    winner = results[0][0]
    st.success(f"**Prediction:** {winner}")

else:
    st.warning("Waiting for image capture...")