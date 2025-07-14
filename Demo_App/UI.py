import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from models.vgg16_lka import VGG16_LKA

# Thay path nếu cần
# MODEL_WEIGHTS = 'D:\\ki_8_9_2025\\dsp301\\best_weights\\best_weights\\best_model_vgg_lka_3_block_8768.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16_LKA(num_classes=14, dropout=0.5)
state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
if 'model' in state:
    state = state['model']
model.load_state_dict(state)
model.eval()
model.to(DEVICE)

# Các label
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 
                  'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
                  'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

# Load CSV data
df = pd.read_csv('data/test_df.csv')

# Preprocessing với normalize (giống app.py)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(img_name):
    """Predict for a single image"""
    try:
        img_path = f'data/images_001/images/{img_name}'
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # Logic mới: Chọn class có xác suất cao nhất
            max_prob = np.max(probs)
            max_class_idx = np.argmax(probs)
            max_class = disease_labels[max_class_idx]
            
            # Chỉ trả "No Finding" khi tất cả xác suất = 0
            if max_prob == 0:
                pred_labels = ['No Finding']
            else:
                pred_labels = [max_class]
        
        # Get ground truth
        gt_diseases = []
        for disease in disease_labels:
            if df.loc[df['Image_Index'] == img_name, disease].values[0] == 1:
                gt_diseases.append(disease)
        gt_labels = gt_diseases if gt_diseases else ['No Finding']
        
        return pred_labels, gt_labels, probs
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return ['Error'], ['Error'], np.zeros(14)

# Test 1000 images đầu tiên
print("Testing first 1000 images...")
print("="*80)

correct_predictions = 0
correct_predictions_without_no_finding = 0
total_predictions = 0
total_predictions_without_no_finding = 0
results = []

for i in range(min(500, len(df))):
    img_name = df['Image_Index'].iloc[i]
    pred_labels, gt_labels, probs = predict_single_image(img_name)
    
    # Check if prediction is correct (exact match)
    is_correct = set(pred_labels) == set(gt_labels)
    if is_correct:
        correct_predictions += 1
    total_predictions += 1
    
    # Check if prediction is correct excluding "No Finding" cases
    if 'No Finding' not in gt_labels:  # Chỉ đếm cases có bệnh
        total_predictions_without_no_finding += 1
        if is_correct:
            correct_predictions_without_no_finding += 1
    
    # Store results
    results.append({
        'image': img_name,
        'predicted': pred_labels,
        'ground_truth': gt_labels,
        'correct': is_correct,
        'max_prob': np.max(probs),
        'max_disease': disease_labels[np.argmax(probs)]
    })
    
    # Print progress every 50 images
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/1000 images...")

# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
accuracy_without_no_finding = correct_predictions_without_no_finding / total_predictions_without_no_finding * 100 if total_predictions_without_no_finding > 0 else 0

print("\n" + "="*80)
print(f"RESULTS SUMMARY:")
print(f"Total images tested: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print("-" * 50)
print(f"Cases with diseases (excluding 'No Finding'): {total_predictions_without_no_finding}")
print(f"Correct predictions (excluding 'No Finding'): {correct_predictions_without_no_finding}")
print(f"Accuracy without 'No Finding': {accuracy_without_no_finding:.2f}%")
print("="*80)

# Show detailed results
print("\nDETAILED RESULTS:")
print("-" * 80)
print(f"{'Image':<20} {'Predicted':<30} {'Ground Truth':<30} {'Status':<10}")
print("-" * 80)

for result in results:
    status = "✓ CORRECT" if result['correct'] else "✗ WRONG"
    pred_str = ', '.join(result['predicted'])
    gt_str = ', '.join(result['ground_truth'])
    print(f"{result['image']:<20} {pred_str:<30} {gt_str:<30} {status:<10}")

# Show some examples of wrong predictions
wrong_predictions = [r for r in results if not r['correct']]
if wrong_predictions:
    print(f"\nEXAMPLES OF WRONG PREDICTIONS (showing first 10):")
    print("-" * 80)
    for i, result in enumerate(wrong_predictions[:10]):
        print(f"{i+1}. {result['image']}")
        print(f"   Predicted: {result['predicted']}")
        print(f"   Ground Truth: {result['ground_truth']}")
        print(f"   Max prob: {result['max_prob']:.3f} for {result['max_disease']}")
        print()

# Show accuracy by disease type
# Thống kê "No Finding" cases
no_finding_correct = 0
no_finding_total = 0
for result in results:
    if 'No Finding' in result['ground_truth']:
        no_finding_total += 1
        if result['correct']:
            no_finding_correct += 1

print(f"\n'NO FINDING' STATISTICS:")
print("-" * 50)
print(f"Total 'No Finding' cases: {no_finding_total}")
print(f"Correct 'No Finding' predictions: {no_finding_correct}")
if no_finding_total > 0:
    print(f"'No Finding' accuracy: {no_finding_correct/no_finding_total*100:.2f}%")

print(f"\nACCURACY BY DISEASE TYPE:")
print("-" * 50)
for disease in disease_labels:
    disease_correct = 0
    disease_total = 0
    for result in results:
        if disease in result['ground_truth']:
            disease_total += 1
            if disease in result['predicted']:
                disease_correct += 1
    
    if disease_total > 0:
        disease_accuracy = disease_correct / disease_total * 100
        print(f"{disease:<20} {disease_correct}/{disease_total} ({disease_accuracy:.1f}%)")
    else:
        print(f"{disease:<20} No cases in test set")

print("\n" + "="*80)
print("TESTING COMPLETED!")
