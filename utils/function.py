import pickle
import h5py
import numpy as np
import cv2
from PIL import Image

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

def convert_h5_to_pkl(h5_path, pkl_path):
    """Converts an H5 model to a PKL model."""
    try:
        model = load_model(h5_path)

        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(model, pkl_file)

        print(f"Converted {h5_path} to {pkl_path}")
        return True
    except Exception as e:
        print(f"Error converting {h5_path} to {pkl_path}: {e}")
        return False

def compute_lbp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    neighbors = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center_pixel = gray_image[y, x]
            binary_string = ''
            for dy, dx in neighbors:
                neighbor_pixel = gray_image[y + dy, x + dx]
                binary_string += '1' if neighbor_pixel >= center_pixel else '0'
            lbp_value = int(binary_string, 2)
            lbp_image[y, x] = lbp_value
    return lbp_image

def load_pretrained_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=base_model.input, outputs=x)

    return feature_extractor

def extract_inception_features(img_path):
    # img = image.load_img(img_path, target_size=(224, 224))
    images = Image.open(img_path).convert("RGB")
    img = images.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = load_pretrained_model().predict(img_array)
    return features.flatten()

def process_image(img_path):
    # img_original = cv2.imread(img_path)
    file_stream = img_path.stream
    file_stream.seek(0) 
    
    # Konversi ke numpy array
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_original is None:
        raise ValueError("Failed to decode image")
    
    lbp_image = compute_lbp(img_original)
    hist_lbp, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256))
    hist_lbp = hist_lbp.astype('float32')
    hist_lbp /= hist_lbp.sum()  # Normalisasi histogram
    
    features_inception = extract_inception_features(img_path)

    combined_features = np.hstack([features_inception, hist_lbp])
    
    return combined_features