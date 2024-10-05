import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from utils import preprocess_image

# Load the saved model (ensure your weight file is in the correct path)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

# Main function to run the Streamlit app
def main():
    st.title('Autism Image Classification')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        # Load the model
        model = load_model()
        
        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Display results
        if predicted_class == 0:
            st.success("The model predicts: Autism is not present")
        else:
            st.warning("The model predicts: Autism is present")

if __name__ == "__main__":
    main()
