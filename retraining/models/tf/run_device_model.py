import keras_cv
import keras_core as keras

import numpy as np
import cv2
from PIL import Image

MODEL_PATH = r"C:\Users\User\Documents\lgbootcamp\project_8\models\deeplab_v3_plus_resnet50_pascalvoc\old_model\segmentation_model.h5"

class ImageSegmenationDevice():
    def __init__(self, model_path=None, model_preset=None, num_classes=None, input_shape=(257, 257),):
        if model_path is None:
            model_path = MODEL_PATH

        self.model = self.load_model(model_path=model_path, model_preset=model_preset, num_classes=num_classes, input_shape=input_shape)
        self.input_shape = input_shape

    def load_model(self, model_path=None, model_preset=None, num_classes=None, input_shape=None):
        if model_path:
            with keras.utils.custom_object_scope({'DeepLabV3Plus': keras_cv.models.DeepLabV3Plus}):
                model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            model = keras_cv.models.DeepLabV3Plus.from_preset(
                model_preset,
                num_classes=num_classes,
                input_shape=input_shape,
            )
        
        return model
    
    def preprocess_image(self, image):
        # image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize = keras_cv.layers.Resizing(height=self.input_shape[0], width=self.input_shape[1])
        image_resized = resize(image)
        return np.array(image_resized, dtype=np.uint8)

    def run_inference(self, image_array):
        image_expanded = keras.ops.expand_dims(image_array, axis=0)
        preds = keras.ops.expand_dims(keras.ops.argmax(self.model(image_expanded), axis=-1), axis=-1).numpy()
        return preds[0, :, :, 0]