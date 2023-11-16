import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cnn import CNN  

# ... (Your existing code)

# Model summary with styles
model_summary = """
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #333;">CNN Model Summary</h3>
        <p style="color: #555;">
            This CNN model consists of five convolutional layers, each followed by a ReLU activation function and a max pooling layer. 
            The output of the convolutional layers is then flattened and passed through three fully connected layers, each followed by a ReLU activation function. 
            The final fully connected layer has a number of outputs equal to the number of classes in the dataset.
        </p>
    </div>
"""

# ... (Your existing code)

def main():
    st.title("Image Classifier")

    # Display model summary
    st.markdown(model_summary, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    # Styles for uploaded image and predicted class
    image_style = "border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"
    predicted_class_style = "font-size: 24px; font-weight: bold; color: #3498db;"

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True, style=image_style)

        try:
            processed_image = preprocess_image(np.array(image))
            predicted_label = predict(processed_image)

            # Display predicted class
            st.markdown(f'<p style="{predicted_class_style}">Predicted Class: {predicted_label}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
