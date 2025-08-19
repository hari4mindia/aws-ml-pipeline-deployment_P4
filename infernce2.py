import json, logging, sys, os, io, base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# must match training norm
NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])

def _build_model(num_classes):
    m = models.resnet18(pretrained=False)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    m.eval()
    return m

# ---- SageMaker handlers ----
def model_fn(model_dir):
    labels_path = os.path.join(model_dir, "classes.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            classes = json.load(f)
        num_classes = len(classes)
    else:
        num_classes = 133
        classes = list(range(num_classes))
    model = _build_model(num_classes)
    state_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.classes = classes
    return model

def input_fn(request_body, request_content_type):
    if request_content_type in ("image/jpeg", "image/png"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return TF(image).unsqueeze(0)
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        if "b64" in payload:
            image = Image.open(io.BytesIO(base64.b64decode(payload["b64"]))).convert("RGB")
            return TF(image).unsqueeze(0)
        if "url" in payload:
            import requests
            r = requests.get(payload["url"], timeout=5)
            r.raise_for_status()
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
            return TF(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(inputs, model):
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().tolist()

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept: {accept}")
