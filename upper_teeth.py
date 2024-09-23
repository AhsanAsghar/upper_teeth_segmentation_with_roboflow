
import json
import cv2
import numpy as np
import os
import sys
import requests
import base64
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
import logging
import torch
from collections import OrderedDict
import detectron2


# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
from PIL import Image


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processing_upper import (
    filter_predictions_by_confidence,
    remove_duplicate_predictions,
    correct_predictions,
    draw_predictions
)

def detect_upper_teeth():

    image_url = ""

    image = cv2.imread(image_url)

    # Save the image temporarily
    temp_image_path = "temp_upper_image.jpg"
    
    cv2.imwrite(temp_image_path, image)

    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass
    

    original_stdout = sys.stdout
    sys.stdout = DummyFile()
    rf = Roboflow(api_key="CDxrYtIlfwTupxOwIJDJ")
    project = rf.workspace().project("segmentation-upper-teeth2")
    model = project.version("2").model
    sys.stdout = original_stdout

    response = model.predict(temp_image_path, confidence=24)
    result = response.json()

    
    # Filter and process predictions
    filtered_predictions = filter_predictions_by_confidence(result['predictions'], 0.42)
    unique_predictions = remove_duplicate_predictions(filtered_predictions, image.shape[1])
    corrected_predictions = correct_predictions(unique_predictions, image.shape[1])
    final_predictions = [pred for pred in corrected_predictions if (11 <= int(pred['class']) <= 18) or (21 <= int(pred['class']) <= 28)]

    # Draw predictions on image
    output_image = draw_predictions(image, final_predictions)

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', output_image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Convert predictions to JSON serializable format
    serializable_predictions = []
    for pred in final_predictions:
        serializable_pred = pred.copy()
        serializable_pred['points'] = [{'x': p['x'], 'y': p['y']} for p in pred['points']]
        serializable_predictions.append(serializable_pred)
    
    
    print("Final_Predictions:", final_predictions)



if __name__ == "__main__":
    # This block will only run when this script is executed directly.
    print("Running directly!")
    detect_upper_teeth()
 
   
    
    
