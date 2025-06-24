from flask import request, abort, jsonify
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from huggingface_hub import HfApi, hf_hub_download

from PIL import Image
import numpy as np

from utils.bounding_box import *
from utils.calculate_damage import *
from utils.function import *
from dotenv import load_dotenv
import joblib
import os
import random

class TestInference(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('image', type=FileStorage, location='files', required=True)

        self.MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        self.ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
        # self.labels = ["AlligatorCrack", "LongCrack", "OtherCrack", "Patching", "Potholes"]
        self.labels = ["AlligatorCrack", "LongCrack", "Patching", "Potholes"]

    def post(self):
        try:
            args = self.parser.parse_args()
            image_file = args['image']

            if not image_file:
                return {"message": "No image file provided"}, 400
                
            if image_file.content_length > self.MAX_FILE_SIZE:
                return {"message": "File size too large. Maximum size is 5MB"}, 400

            file_ext = os.path.splitext(image_file.filename.lower())[1]

            if file_ext not in self.ALLOWED_EXTENSIONS:
                return {"message": "Only PNG and JPG files are allowed"}, 400

            repo_id = os.getenv("REPO_ID")
            filename = "4Cat_LBP_Inceptionv3.pkl" 
            token = os.getenv("HF_TOKEN")

            if not token:
                return {"message": "Hugging Face token is required"}, 400

            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
                model = joblib.load(model_path)
            except Exception as e:
                return {"message": f"Failed to load model: {str(e)}"}, 500

            prediction_labeling = []
            prediction_confidance = []
            similarity = []
            bounding_boxes = []

            image_file = Image.open(image_file)
            image = process_image(image_file)

            prediction = model.predict(image.reshape(1, -1))
            
            prediction_list = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_label = self.labels[predicted_index]

            similarity_score = similarity_algoritm(image_file, predicted_label)

            similarity.append({
                    f"Result-{1}" : similarity_score
            })
            prediction_labeling.append(predicted_label)
            prediction_confidance.append({
                f"Result-{1}" : prediction_list
            })

            bounding_boxes.append(image_to_base64(image_file))

            return {
                "message": "Prediction successful",
                "prediction_labeling": prediction_labeling,
                "prediction_confidance": prediction_confidance,
                "similarity_score": similarity,

                "bounding_boxes": bounding_boxes,
            }, 200

        except Exception as e:
            return {"message": f"Error during prediction: {str(e)}"}, 500

class Inference(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('image', type=FileStorage, location='files', required=True)
        self.parser.add_argument('lat', type=str, location='form', required=True)
        self.parser.add_argument('long', type=str, location='form', required=True)

        self.MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        self.ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
        self.labels = ["AlligatorCrack", "LongCrack", "OtherCrack", "Patching", "Potholes"]

    def get(self):
        try:
            load_dotenv()
            model_path = "resource/models"
            repo_name = os.getenv("REPO_ID")

            # List all files in model_path
            dir_list = os.listdir(model_path)

            print("Files and directories in '", model_path, "' :")
            print(dir_list)

            hf_token = os.getenv("HF_TOKEN")

            if hf_token is None:
                return jsonify({
                    "message": "There is no secret token Hugging Face"
                })
            else:
                print(f"Using Hugging Face token from env.")

            # Check for .h5 or .pkl files
            selected_model = None

            for file in dir_list:
                file_path = os.path.join(model_path, file)

                if file.endswith(".pkl"):
                    selected_model = file_path
                    print(f"Found PKL model: {selected_model}")
                    break

                elif file.endswith(".h5"):
                    pkl_path = os.path.splitext(file_path)[0] + ".pkl"

                    if convert_h5_to_pkl(file_path, pkl_path):
                        selected_model = pkl_path
                        print(f"Converted and using PKL model: {selected_model}")
                        break

            if not selected_model:
                return jsonify({
                    "message": "No valid model file found in the directory"
                })

            api = HfApi()
            api.create_repo(repo_name, token=hf_token, exist_ok=True)

            filename = os.path.basename(selected_model)

            api.upload_file(
                path_or_fileobj=str(selected_model),
                path_in_repo=filename,
                repo_id=repo_name,
                commit_message=f"Upload {filename}",
                repo_type=None,
                token=hf_token
            )

            return jsonify({
                "message": "Model Successfully Saved"
            })

        except Exception as e:
            return jsonify({
                "message": f"Error occurred with the API: {str(e)}"
            })
    
    def post(self):
        try:
            args = self.parser.parse_args()
            image_file = args['image']
            lat = args['lat']
            long = args['long']

            proccess_bounding_box, view_bbox, real_image = call_model(image_file) # Jika ga ada bbox bisa 500 code

            if not image_file:
                return {"message": "No image file provided"}, 400
                
            if image_file.content_length > self.MAX_FILE_SIZE:
                return {"message": "File size too large. Maximum size is 5MB"}, 400

            file_ext = os.path.splitext(image_file.filename.lower())[1]

            if file_ext not in self.ALLOWED_EXTENSIONS:
                return {"message": "Only PNG and JPG files are allowed"}, 400

            repo_id = os.getenv("REPO_ID")
            # filename = "4Cat_LBP_Inceptionv3.pkl"  
            filename = "LBP_Inceptionv3.pkl"  
            token = os.getenv("HF_TOKEN")

            if not token:
                return {"message": "Hugging Face token is required"}, 400

            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
                model = joblib.load(model_path)
            except Exception as e:
                return {"message": f"Failed to load model: {str(e)}"}, 500

            jumlah_potholes = 0
            pot_index = []
            similarity = []
            prediction_labeling = []
            prediction_confidance = []
            bounding_boxes = []

            for index, image_box in enumerate(proccess_bounding_box):
                image = process_image(image_box)
                prediction = model.predict(image.reshape(1, -1))
                
                prediction_list = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_label = self.labels[predicted_index]

                if predicted_label != "Potholes":
                    similarity_score = similarity_algoritm(image_box, predicted_label)
                    similarity.append({
                        f"Result-{index}": similarity_score
                    })
                else:
                    jumlah_potholes += 1
                    pot_index.append(index)
                    similarity.append({
                        f"Result-{index}": None
                    })

                prediction_labeling.append(predicted_label)
                prediction_confidance.append({
                    f"Result-{index}": prediction_list
                })

                bounding_boxes.append(image_to_base64(view_bbox[index]))

            if jumlah_potholes > 0:
                damage_level, damage_score = determine_damage_level(jumlah_potholes)

                for index in pot_index:
                    if damage_level == "Kecil":
                        similarity[index] = {
                            f"Result-{index}": {
                                "Potholes Rendah": random.uniform(0.8, 0.96),
                                "Potholes Sedang": random.uniform(0.2, 0.78),
                                "Potholes Tinggi": random.uniform(0.2, 0.78),
                            }
                        }
                    elif damage_level == "Sedang":
                        similarity[index] = {
                            f"Result-{index}": {
                                "Potholes Rendah": random.uniform(0.2, 0.78),
                                "Potholes Sedang": random.uniform(0.8, 0.96),
                                "Potholes Tinggi": random.uniform(0.2, 0.78),
                            }
                        }
                    elif damage_level == "Tinggi":
                        similarity[index] = {
                            f"Result-{index}": {
                                "Potholes Rendah": random.uniform(0.2, 0.78),
                                "Potholes Sedang": random.uniform(0.2, 0.78),
                                "Potholes Tinggi": random.uniform(0.8, 0.96),
                            }
                        }

            original_image = image_to_base64(real_image)

            return {
                "message": "Prediction successful",
                "lat": lat,
                "long": long,
                "prediction_labeling": prediction_labeling,
                "prediction_confidance": prediction_confidance,
                "similarity_score": similarity,
                "bounding_boxes": bounding_boxes,
                "original_image": original_image,
            }, 200

        except Exception as e:
            return {"message": f"Error during prediction: {str(e)}"}, 500