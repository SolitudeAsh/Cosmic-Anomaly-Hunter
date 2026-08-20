import torch
import numpy as np
import cv2

def grad_cam(model, image_tensor, target_class=None):
    model.eval()
    # locate last conv layer (for ResNet it's 'layer4' or 'conv5')
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'conv' in name:
            last_conv = module
    if last_conv is None:
        raise ValueError("No conv layer found")

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = last_conv.register_forward_hook(forward_hook)
    handle_b = last_conv.register_backward_hook(backward_hook)

    image_tensor.requires_grad_(True)
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax().item()
    model.zero_grad()
    target = output[0, target_class]
    target.backward()

    handle_f.remove()
    handle_b.remove()

    acts = activations[0].detach().cpu().numpy()[0]   # (C, H, W)
    grads = gradients[0].detach().cpu().numpy()[0]    # (C, H, W)
    weights = np.mean(grads, axis=(1, 2))             # (C,)

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam