import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

MODEL_PATH = r"C:\Users\User\Documents\lgbootcamp\project_8\models\tflite_model\1.tflite"

class ImageSegmenationBase():
    def __init__(self, model_path=None):
        self.model_path = model_path
        
        if model_path is None:
            self.model_path = MODEL_PATH

        self.model = self.load_model(self.model_path)
        self.input_shape = self.model.get_input_details()[0]['shape'][1:3] # (257, 257)

    def load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize(self.input_shape, Image.Resampling.LANCZOS)
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    
    def run_inference(self, image):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        self.model.set_tensor(input_index, image)
        self.model.invoke()

        output = self.model.get_tensor(output_index)
        output = np.squeeze(output)  # Remove batch dimension

        # Convert one-hot encoded output to class labels
        if len(output.shape) == 3 and output.shape[-1] > 1:
            output = np.argmax(output, axis=-1)

        return output