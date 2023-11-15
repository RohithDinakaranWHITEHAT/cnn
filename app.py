import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cnn import CNN  # Assuming you have a separate file for your model definition

# Load the trained model
model = CNN(num_classes=36, dropout_rate=0.2)
model.load_state_dict(torch.load("rohithdi_ssnehaba_assignment2_part3.h5", map_location=torch.device('cpu')))
model.eval()

# Include the following function for preprocessing:
def preprocess_image(image):
    # Convert the image to PIL Image
    image = Image.fromarray(image)
    
    # Apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    return image

# Map numerical class to original label
label_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b',
    12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h',
    18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n',
    24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
}

# CSS style for bold and larger size
css_style = """
    p.predicted-class {
        font-size: 30px;
        font-weight: bold;
        animation: textZoom 1s ease;
        text-align: center;
    }

    @keyframes textZoom {
        from { font-size: 30px; }
        to { font-size: 16px; }
    }
"""

def predict(image):
    with torch.no_grad():
        output = model(image)

    _, predicted_class = torch.max(output, 1)
    predicted_class = int(predicted_class)

    original_label = label_mapping.get(predicted_class, 'Unknown')

    return original_label

def main():
    st.title("Image Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    # Add CSS style for animation
    st.markdown(f'<style>{css_style}</style>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            processed_image = preprocess_image(np.array(image))
            predicted_label = predict(processed_image)

            # Apply animation style
            st.markdown(f'<p class="predicted-class">Predicted Class: {predicted_label}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
