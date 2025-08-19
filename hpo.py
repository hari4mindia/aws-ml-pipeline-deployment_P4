import os, sys, argparse, logging, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ---------- data loaders ----------
def get_data_loaders(data_dir, batch_size):
    # Expect SageMaker channels: train/, valid/, test/ inside data_dir
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    # standard ImageNet-ish aug/normalization
    norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = torchvision.datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_ds  = torchvision.datasets.ImageFolder(test_dir,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader, len(train_ds.classes)

# ---------- model ----------
def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False  # transfer learning head-only
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

# ---------- train / eval ----------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects, n = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            b = inputs.size(0)
            running_loss += loss.item() * b
            running_corrects += torch.sum(preds == labels).item()
            n += b
    return running_loss / max(n, 1), running_corrects / max(n, 1)

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=5, patience=2):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad = 0
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, n = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            b = inputs.size(0)
            running_loss += loss.item() * b
            running_corrects += torch.sum(preds == labels).item()
            n += b
        train_loss = running_loss / n
        train_acc  = running_corrects / n

        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("Early stopping")
                break

    model.load_state_dict(best_wts)
    return model

# ---------- main ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, valid_loader, test_loader, num_classes = get_data_loaders(args.data, args.batch_size)
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    model = train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=args.epochs, patience=2)

    # REQUIRED for HPO metric extraction: print this EXACT format
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Testing Loss: {test_loss:.4f}")     # <- notebook Regex looks for this
    logger.info(f"Testing Accuracy: {test_acc:.4f}")

    # save model for SageMaker
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # HPO search space keys must match the tuner in the notebook
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--epochs", type=int,       default=5)

    # SageMaker channels / dirs
    parser.add_argument("--data", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    args = parser.parse_args()
    logger.info(args)
    main(args)
