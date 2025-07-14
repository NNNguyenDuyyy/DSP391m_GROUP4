import torch
import numpy as np
import cv2
from PIL import Image

def compute_gradcam(model, img_tensor, class_idx, device):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    grad = None
    activation = None

    def save_grad(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    def save_activation(module, input, output):
        nonlocal activation
        activation = output.detach()

    handle_fwd = model.lka5.register_forward_hook(save_activation)
    handle_bwd = model.lka5.register_backward_hook(save_grad)

    output = model(img_tensor)
    pred = torch.sigmoid(output)
    model.zero_grad()
    pred[:, class_idx].backward()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_fwd.remove()
    handle_bwd.remove()
    return cam

def overlay_gradcam_on_image(img_pil, cam):
    img = np.array(img_pil.resize((224, 224)).convert('RGB'))
    # Ensure cam is 2D and resize to (224, 224)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Ensure both are uint8 and same shape
    if heatmap.shape != img.shape:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return Image.fromarray(overlay)

