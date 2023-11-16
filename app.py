import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cnn import CNN  

model = CNN(num_classes=36, dropout_rate=0.2)
model.load_state_dict(torch.load("rohithdi_ssnehaba_assignment2_part3.h5", map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

label_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b',
    12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h',
    18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n',
    24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
}

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

    div.model-summary {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    div.model-summary h3 {
        color: #333;
    }

    div.model-summary p {
        color: #555;
    }
"""

model_summary = """
    <div class="model-summary">
        <h3>Model Summary</h3>
        <p><strong>CNN Architecture:</strong><br>
            This CNN model consists of five convolutional layers, each followed by a ReLU activation function and a max pooling layer.
            The output of the convolutional layers is then flattened and passed through three fully connected layers, each followed by a ReLU activation function.
            The final fully connected layer has a number of outputs equal to the number of classes in the dataset.
        </p>
        <p><strong>Training Process:</strong><br>
            The model was trained using early stopping for 400 epochs with a batch size of 16 and an Adagrad optimizer with a learning rate of 0.001.
            The model achieved an accuracy of 89.12% on the test set.
        </p>
    </div>
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

    st.markdown(f'<style>{css_style}</style>', unsafe_allow_html=True)
    st.markdown(model_summary, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            processed_image = preprocess_image(np.array(image))
            predicted_label = predict(processed_image)

            st.markdown(f'<p class="predicted-class">Predicted Class: {predicted_label}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
