import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import unicodedata
from torch.nn.utils.rnn import pack_padded_sequence
import pytesseract
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

# Provided Vietnamese alphabet
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'

# Step 1: Prepare dataset JSON from CSV
def create_dataset_json(csv_path, img_dir, json_path, split='train'):
    df = pd.read_csv(csv_path, sep='\t')
    data_list = []
    for _, row in df.iterrows():
        img_name = f"{row['id']}.png"
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            data_list.append({
                'name': img_name,
                'text': row['label']
            })
    
    dataset = {
        'abc': ALPHABET,
        split: data_list
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Created {json_path} with alphabet length {len(ALPHABET)} and {len(data_list)} samples.")

# CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=32, nc=1, nh=256):
        super(CRNN, self).__init__()
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # RNN layers
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, num_layers=2, batch_first=True),
            nn.Linear(nh * 2, num_classes)
        )

    def forward(self, x):
        conv = self.cnn(x)
        bs, c, h, w = conv.size()
        assert h == 1, "Height must be 1 after CNN"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn[0](conv)
        out = self.rnn[1](rnn_out)
        return out

# Custom Dataset
class OCRDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, split='train'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.alphabet = data['abc']
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}  # +1 for blank
        self.samples = data[split]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['name'])
        image = Image.open(img_path).convert('L')
        label = sample['text']
        label_encoded = [self.char_to_idx.get(char, 0) for char in label]  # Use 0 for unknown chars
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_encoded), len(label_encoded)

# Collate function for variable lengths
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)
    return images, labels, lengths

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long).to(device)
            log_probs = log_probs.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss, cer = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, CER: {cer:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'crnn_epoch_{epoch+1}.pth')

# Evaluation with CER and pytesseract comparison
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_cer = 0
    total_samples = 0
    with torch.no_grad():
        for images, targets, lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long).to(device)
            log_probs = log_probs.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, lengths)
            total_loss += loss.item()
            
            preds = log_probs.argmax(2).cpu().numpy().T
            pred_strs = []
            for i in range(preds.shape[0]):
                pred = ''
                prev = -1
                for j in preds[i]:
                    if j != 0 and j != prev:
                        pred += ALPHABET[j - 1] if j - 1 < len(ALPHABET) else ''
                    prev = j
                pred_strs.append(pred)
            
            gt_idx = 0
            gt_strs = []
            for length in lengths.cpu().numpy():
                gt = ''.join([ALPHABET[idx - 1] for idx in targets[gt_idx:gt_idx + length].cpu().numpy() if idx > 0])
                gt_strs.append(gt)
                gt_idx += length
            
            for pred, gt in zip(pred_strs, gt_strs):
                total_cer += levenshtein_distance(pred, gt) / max(len(gt), 1)
            total_samples += len(gt_strs)
            
            # Pytesseract comparison
            for img, gt in zip(images, gt_strs):
                img_pil = transforms.ToPILImage()(img.cpu())
                tess_text = pytesseract.image_to_string(img_pil, lang='vie', config='--psm 6')
                tess_cer = levenshtein_distance(tess_text.strip(), gt) / max(len(gt), 1)
                print(f"Pytesseract CER: {tess_cer:.4f} for sample")

    return total_loss / len(loader), total_cer / total_samples

# Main
if __name__ == "__main__":
    data_dir = './data'
    level = 'paragraph'
    
    # Create JSON for datasets
    create_dataset_json(os.path.join(data_dir, f'train_{level}.csv'), os.path.join(data_dir, f'train_{level}'), 'train.json', 'train')
    create_dataset_json(os.path.join(data_dir, f'validation_{level}.csv'), os.path.join(data_dir, f'validation_{level}'), 'val.json', 'train')
    create_dataset_json(os.path.join(data_dir, f'test_{level}.csv'), os.path.join(data_dir, f'test_{level}'), 'test.json', 'test')

    # Number of classes: alphabet length + 1 for blank
    num_classes = len(ALPHABET) + 1

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((32, None)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets
    train_dataset = OCRDataset('train.json', os.path.join(data_dir, f'train_{level}'), transform)
    val_dataset = OCRDataset('val.json', os.path.join(data_dir, f'validation_{level}'), transform)
    test_dataset = OCRDataset('test.json', os.path.join(data_dir, f'test_{level}'), transform)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model
    model = CRNN(num_classes)

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device=device)

    # Test
    _, test_cer = evaluate(model, test_loader, nn.CTCLoss(), device)
    print(f"Test CER: {test_cer:.4f}")