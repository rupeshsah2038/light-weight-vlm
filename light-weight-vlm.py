import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import jsonlines
import os
import glob
import numpy as np
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import csv
from pathlib import Path

# Create directories for logs and results
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

class LightweightVLM(nn.Module):
    """A lightweight Vision-Language Model (<1B parameters) for modality and location classification."""
    def __init__(self, vision_dim=512, text_dim=768, num_classes_modality=2, num_classes_location=6):
        super(LightweightVLM, self).__init__()
        # Simple CNN for visual feature extraction
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, vision_dim)
        )
        # LSTM-based text encoder
        self.text_embedding = nn.Embedding(30522, 256)
        self.text_lstm = nn.LSTM(256, text_dim, num_layers=2, batch_first=True)
        self.text_linear = nn.Linear(text_dim, text_dim)
        # Classification heads
        self.modality_head = nn.Linear(vision_dim + text_dim, num_classes_modality)
        self.location_head = nn.Linear(vision_dim + text_dim, num_classes_location)

    def forward(self, images, texts):
        vision_features = self.vision_encoder(images)
        embedded_texts = self.text_embedding(texts)
        lstm_outputs, _ = self.text_lstm(embedded_texts)
        text_features = self.text_linear(lstm_outputs[:, -1, :])
        combined_features = torch.cat((vision_features, text_features), dim=1)
        modality_logits = self.modality_head(combined_features)
        location_logits = self.location_head(combined_features)
        return modality_logits, location_logits

class MedPixDataset(Dataset):
    """Custom Dataset for MedPix 2.0 images and text with caching for faster loading."""
    def __init__(self, data_jsonl_files, desc_jsonl_files, image_dir, tokenizer, processor, cache_dir='cache'):
        self.data = []
        with jsonlines.open(data_jsonl_files, 'r') as reader:
            for obj in reader:
                self.data.append(obj)
        
        self.desc = []
        with jsonlines.open(desc_jsonl_files, 'r') as reader:
            for obj in reader:
                self.desc.append(obj)
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Label mappings
        self.modality_map = {'CT': 0, 'MR': 1}
        self.location_map = {'Abdomen': 0, 'Head': 1, 'RUS': 2, 'Thorax': 3, 'Spine and Muscles': 4, 'Reproductive and Urinary System': 5}
        self.cache = {}
        self._build_cache()

    def _build_cache(self):
        """Populate in-memory cache dictionary with preprocessed samples if available."""
        for idx in range(len(self.desc)):
            cache_file = os.path.join(self.cache_dir, f'item_{idx}.pt')
            if os.path.exists(cache_file):
                self.cache[idx] = cache_file

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        # Return from cache if available
        if idx in self.cache:
            return torch.load(self.cache[idx])

        desc = self.desc[idx]
        image_path = os.path.join(self.image_dir, desc['image'] + '.png')
        image = Image.open(image_path).convert('RGB')
        image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # Extract associated text (history + caption)
        case = next((item for item in self.data if item['U_id'] == desc['U_id']), None)
        history = case['Case']['History'] if case and case.get('Case') and case['Case'].get('History') else ""
        caption = desc['Description']['Caption'] if desc.get('Description') and desc['Description'].get('Caption') else ""
        text = f"{history} {caption}".strip() or "No description available"

        # Tokenize text
        text_tokens = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True).input_ids.squeeze()

        # Get labels
        modality_label = self.modality_map[desc['Type']]
        location_label = self.location_map[desc['Location Category']]

        # Cache result for reuse
        cache_file = os.path.join(self.cache_dir, f'item_{idx}.pt')
        torch.save((image, text_tokens, modality_label, location_label), cache_file)
        self.cache[idx] = cache_file

        return image, text_tokens, modality_label, location_label

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss with label smoothing for stable training."""
    def __init__(self, alpha=0.5, temperature=2.0, label_smoothing=0.1):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        student_modality, student_location = student_logits
        teacher_modality, teacher_location = teacher_logits
        # Combine supervised and soft distillation losses
        loss_modality = self.ce_loss(student_modality, labels[0]) + self.alpha * self.kl_div(
            torch.log_softmax(student_modality / self.temperature, dim=1),
            torch.softmax(teacher_modality / self.temperature, dim=1)
        )
        loss_location = self.ce_loss(student_location, labels[1]) + self.alpha * self.kl_div(
            torch.log_softmax(student_location / self.temperature, dim=1),
            torch.softmax(teacher_location / self.temperature, dim=1)
        )
        return loss_modality + loss_location

class TeacherModel(nn.Module):
    """Teacher model that combines CLIP and DistilBERT outputs for supervision."""
    def __init__(self, clip_model, distilbert_model):
        super(TeacherModel, self).__init__()
        self.clip_model = clip_model
        self.distilbert_model = distilbert_model
        self.modality_head = nn.Linear(512 + 768, 2)
        self.location_head = nn.Linear(512 + 768, 6)

    def forward(self, images, texts):
        vision_outputs = self.clip_model.get_image_features(images)
        text_outputs = self.distilbert_model(texts).last_hidden_state[:, 0, :]
        combined_features = torch.cat((vision_outputs, text_outputs), dim=1)
        modality_logits = self.modality_head(combined_features)
        location_logits = self.location_head(combined_features)
        return modality_logits, location_logits

def train_and_optimize(model, teacher_model, train_loader, val_loader, epochs=20, accum_steps=4):
    """Train and optimize the lightweight model with knowledge distillation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    teacher_model.to(device).eval()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DistillationLoss()
    scaler = GradScaler()

    # Freeze some layers initially for stability
    for param in model.vision_encoder[:4].parameters():
        param.requires_grad = False
    for param in model.text_embedding.parameters():
        param.requires_grad = False

    log_file = 'logs/training_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_modality_acc', 'val_location_acc',
                         'modality_precision', 'modality_recall', 'modality_f1',
                         'location_precision', 'location_recall', 'location_f1'])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        if epoch == 5:  # Unfreeze layers after 5 epochs
            for param in model.vision_encoder.parameters():
                param.requires_grad = True
            for param in model.text_embedding.parameters():
                param.requires_grad = True

        for i, (images, texts, modality_labels, location_labels) in enumerate(train_loader):
            images, texts = images.to(device), texts.to(device)
            modality_labels, location_labels = modality_labels.to(device), location_labels.to(device)

            with torch.no_grad():
                teacher_modality, teacher_location = teacher_model(images, texts)

            with autocast('cuda'):
                student_modality, student_location = model(images, texts)
                loss = criterion((student_modality, student_location), (teacher_modality, teacher_location), (modality_labels, location_labels))
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        scheduler.step()

        # Validation phase
        model.eval()
        correct_modality, correct_location, total = 0, 0, 0
        modality_preds, location_preds = [], []
        modality_labels_all, location_labels_all = [], []

        with torch.no_grad():
            for images, texts, modality_labels, location_labels in val_loader:
                images, texts = images.to(device), texts.to(device)
                modality_labels, location_labels = modality_labels.to(device), location_labels.to(device)
                with autocast('cuda'):
                    modality_logits, location_logits = model(images, texts)
                _, pred_modality = torch.max(modality_logits, 1)
                _, pred_location = torch.max(location_logits, 1)
                correct_modality += (pred_modality == modality_labels).sum().item()
                correct_location += (pred_location == location_labels).sum().item()
                total += modality_labels.size(0)
                modality_preds.extend(pred_modality.cpu().numpy())
                location_preds.extend(pred_location.cpu().numpy())
                modality_labels_all.extend(modality_labels.cpu().numpy())
                location_labels_all.extend(location_labels.cpu().numpy())

        modality_acc = correct_modality / total
        location_acc = correct_location / total
        modality_metrics = precision_recall_fscore_support(modality_labels_all, modality_preds, average='macro', zero_division=1)
        location_metrics = precision_recall_fscore_support(location_labels_all, location_preds, average='macro', zero_division=1)

        # Log results to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, total_loss / len(train_loader), modality_acc, location_acc,
                             modality_metrics[0], modality_metrics[1], modality_metrics[2],
                             location_metrics[0], location_metrics[1], location_metrics[2]])

        # Save confusion matrices for analysis
        np.savetxt(f'logs/confusion_modality_epoch_{epoch+1}.csv', confusion_matrix(modality_labels_all, modality_preds), delimiter=',', fmt='%d')
        np.savetxt(f'logs/confusion_location_epoch_{epoch+1}.csv', confusion_matrix(location_labels_all, location_preds), delimiter=',', fmt='%d')

        print(f'Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Modality Acc: {modality_acc:.4f}, Location Acc: {location_acc:.4f}')

    # Pruning for model compression
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=0.3)

    # Remove pruning reparametrization
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass

    # Apply dynamic quantization for further size reduction
    model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    return model

if __name__ == '__main__':
    # Load pretrained teacher model components
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    distilbert_model = AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    teacher_model = TeacherModel(clip_model, distilbert_model)

    # Prepare datasets
    train_dataset = MedPixDataset('MedPix-2-0/splitted_dataset/data_train.jsonl', 'MedPix-2-0/splitted_dataset/descriptions_train.jsonl', 'MedPix-2-0/images', tokenizer, processor)
    val_dataset = MedPixDataset('MedPix-2-0/splitted_dataset/data_dev.jsonl', 'MedPix-2-0/splitted_dataset/descriptions_dev.jsonl', 'MedPix-2-0/images', tokenizer, processor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # Train and optimize lightweight model
    model = LightweightVLM()
    optimized_model = train_and_optimize(model, teacher_model, train_loader, val_loader)

    # Save the final model
    torch.save(optimized_model.state_dict(), 'results/lightweight_vlm.pth')
