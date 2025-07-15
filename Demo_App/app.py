import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import math

from models.vgg16_lka import VGG16_LKA
from utils.gradcam_utils import compute_gradcam, overlay_gradcam_on_image
from utils.data_utils import load_labels, get_ground_truth

# --- CONFIG ---
DISEASE_LABELS = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
                  'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
MODEL_WEIGHTS = os.path.join('best_weights', 'best_weights', 'best_model_vgg_lka_3_block_8818.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGES_DIR = os.path.join('demo_imgs', 'random_images')
LABELS_CSV = os.path.join('demo_imgs', 'random_img.csv')

# --- LOAD CSV DATA ---
import pandas as pd
df = pd.read_csv(LABELS_CSV)

# --- GET ALL IMAGES ---
# Get images in the same order as CSV
all_image_files = df['Image_Index'].tolist()
print(f"Found {len(all_image_files)} images from CSV")

# --- BATCHING ---
BATCH_SIZE = 100
num_batches = max(1, math.ceil(len(all_image_files) / BATCH_SIZE))
batch_labels = [f"Images {i*BATCH_SIZE+1}-{min((i+1)*BATCH_SIZE, len(all_image_files))}" for i in range(num_batches)]

def get_images_for_batch(batch_idx):
    start = batch_idx * BATCH_SIZE
    end = min((batch_idx + 1) * BATCH_SIZE, len(all_image_files))
    batch_images = all_image_files[start:end]
    return batch_images if batch_images else ["No images in this batch"]

# --- IMAGE PREPROCESS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
model = VGG16_LKA(num_classes=len(DISEASE_LABELS), dropout=0.5)
state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
if 'model' in state:
    state = state['model']
model.load_state_dict(state)
model.eval()
model.to(DEVICE)

def predict(image_file):
    threshold = 0.4  # Thêm ngưỡng xác suất
    # 1. Load image
    if isinstance(image_file, str):
        img_path = os.path.join(IMAGES_DIR, image_file)
        img_pil = Image.open(img_path).convert('RGB')
        image_name = image_file
    else:
        img_pil = Image.open(image_file).convert('RGB')
        image_name = os.path.basename(image_file.name)
    img_tensor = transform(img_pil)

    # 2. Model prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(DEVICE))
        probs = torch.sigmoid(output).cpu().numpy()[0]
        
        # Debug: In ra tất cả xác suất
        print(f"\nPrediction probabilities for {image_name}:")
        for i, (disease, prob) in enumerate(zip(DISEASE_LABELS, probs)):
            print(f"{i+1:2d}. {disease:<20} {prob:.4f}")
        
        # Chọn tất cả class có xác suất > threshold
        pred_indices = [i for i, prob in enumerate(probs) if prob > threshold]
        if pred_indices:
            pred_labels = [DISEASE_LABELS[i] for i in pred_indices]
            pred_labels_str = ', '.join(pred_labels)
        else:
            pred_labels_str = 'No Finding'
        print(f"Predicted: {pred_labels_str}")

    # 3. Ground truth - Get from CSV with disease columns
    gt_diseases = []
    for disease in DISEASE_LABELS:
        if df.loc[df['Image_Index'] == image_name, disease].values[0] == 1:
            gt_diseases.append(disease)
    gt_labels = gt_diseases if gt_diseases else ['No Finding']
    gt_labels_str = ', '.join(gt_labels)

    # 4. GradCAM cho class có xác suất cao nhất (chỉ vẽ nếu có dự đoán)
    if pred_labels_str != 'No Finding':
        if len(probs) > 0:
            class_idx = int(np.argmax(probs))
        else:
            class_idx = 0
        cam = compute_gradcam(model, img_tensor, class_idx, DEVICE)
        gradcam_img = overlay_gradcam_on_image(img_pil, cam)
    else:
        gradcam_img = None

    return img_pil, pred_labels_str, gt_labels_str, gradcam_img

# --- GRADIO UI ---
def gradio_interface(image_file):
    return predict(image_file)

with gr.Blocks() as demo:
    gr.Markdown("# Chest X-ray Diagnosis with GradCAM\nSelect a batch and image to see model predictions and GradCAM visualization.")
    batch_selector = gr.Dropdown(choices=batch_labels, value=batch_labels[0], label="Select Image Batch")
    image_selector = gr.Radio(choices=get_images_for_batch(0), label="Select one of these images below")
    
    def update_image_choices(selected_batch):
        batch_idx = batch_labels.index(selected_batch)
        return gr.update(choices=get_images_for_batch(batch_idx), value=None)

    batch_selector.change(update_image_choices, inputs=batch_selector, outputs=image_selector)

    output_img = gr.Image(type="pil", label="Selected Image")
    output_pred = gr.Textbox(label="Predicted Labels")
    output_gt = gr.Textbox(label="Ground Truth Labels")
    output_gradcam = gr.Image(type="pil", label="GradCAM Heatmap")

    image_selector.change(gradio_interface, inputs=image_selector, outputs=[output_img, output_pred, output_gt, output_gradcam])

if __name__ == "__main__":
    demo.launch()