import pytest
import torch
from PIL import Image
import numpy as np
import os




MODEL_PATH = 'outputs/counterfeit_detector.pth'

from app import get_trained_model, preprocess_image, CLASS_NAMES

MODEL_PATH = 'outputs/counterfeit_detector.pth'


@pytest.fixture
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    return get_trained_model()
@pytest.fixture
def sample_image():
    img = Image.new('RGB', (300, 300), color = 'red')
    return img

def test_model_loading(model):
    assert model is not None
    assert model.classifier[1].out_features == len(CLASS_NAMES)

def test_preprocessing(sample_image):
    dummy_path = "outputs/dummy_test_image.jpg"
    sample_image.save(dummy_path)
    
    tensor, original_img = preprocess_image(dummy_path)
    
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(original_img, np.ndarray)
    
    os.remove(dummy_path)

def test_prediction_pipeline(model):
    img = Image.new('RGB', (224, 224), color = 'blue')
    dummy_path = "outputs/dummy_integration_test.jpg"
    img.save(dummy_path)

    input_tensor, _ = preprocess_image(dummy_path)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        
    assert pred_idx.item() in [0, 1]
    
    os.remove(dummy_path) 