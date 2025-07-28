import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.quantization import quantize_dynamic
import cv2
import numpy as np
from PIL import Image
import os
import time

MODEL_PATH = 'outputs/counterfeit_detector.pth'
TEST_IMAGE_PATH = r'C:\Users\Prassan Katiyar\Desktop\PROJECTS\counterfeit-detection-project\test\0177_jpg.rf.bcaa51637b00f37c8f7d54976a37c0af.jpg'
CLASS_NAMES = ['fake', 'real'] 
OUTPUT_DIR = 'outputs'

def get_trained_model():
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda module, input, output: setattr(self, 'activations', output))
        target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: setattr(self, 'gradients', grad_out[0]))

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        one_hot_output = torch.zeros((1, output.size()[-1]))
        one_hot_output[0][class_idx] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        guided_gradients = self.gradients.data.numpy()[0]
        target_activations = self.activations.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))
        heatmap = np.zeros(target_activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * target_activations[i, :, :]
            
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap, torch.argmax(output).item()

def overlay_heatmap(heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlaid_image = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
    return overlaid_image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), np.array(image)

def benchmark_model(model, input_tensor, model_name="Model"):
    print(f"\n--- Benchmarking {model_name} ---")
    
    torch.save(model.state_dict(), "outputs/temp_model.pth")
    model_size = os.path.getsize("outputs/temp_model.pth") / (1024 * 1024)
    os.remove("outputs/temp_model.pth")
    print(f"Size on Disk: {model_size:.2f} MB")
    
    with torch.no_grad():
        for _ in range(5): _ = model(input_tensor) # Warm-up
        start_time = time.time()
        for _ in range(20): _ = model(input_tensor)
        end_time = time.time()
        
    avg_inference_time = (end_time - start_time) / 20 * 1000
    print(f"Average Inference Speed: {avg_inference_time:.2f} ms")


if __name__ == '__main__':
    input_tensor, original_image = preprocess_image(TEST_IMAGE_PATH)

    model_fp32 = get_trained_model()
    benchmark_model(model_fp32, input_tensor, "FP32 (Original)")
    
    model_quantized = quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
    benchmark_model(model_quantized, input_tensor, "INT8 (Quantized)")
    
    print("\n--- Running Prediction with Grad-CAM ---")
    grad_cam = GradCAM(model=model_fp32, target_layer=model_fp32.features)
    heatmap, predicted_idx = grad_cam.generate_heatmap(input_tensor, class_idx=None)
    
    predicted_class = CLASS_NAMES[predicted_idx]
    print(f"Prediction: {predicted_class.upper()}")
    
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    grad_cam_image = overlay_heatmap(heatmap, original_image_bgr)
    output_path = os.path.join(OUTPUT_DIR, 'grad_cam_result.jpg')
    cv2.imwrite(output_path, grad_cam_image)
    
    print(f"Grad-CAM visualization saved to: {output_path}")