from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to the format expected by the model.
    
    Args:
        image (PIL.Image): The input image to preprocess.
        
    Returns:
        np.ndarray: The preprocessed image suitable for model prediction.
    """
    # ResNet50V2 expects images of size 224x224
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image
    return image
