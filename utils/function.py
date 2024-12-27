import pickle
import h5py
from tensorflow.keras.models import load_model

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