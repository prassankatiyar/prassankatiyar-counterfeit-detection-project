import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

MODEL_SAVE_PATH = 'outputs/counterfeit_detector.pth'
NUM_EPOCHS = 5
BATCH_SIZE = 16

os.makedirs('outputs', exist_ok=True)

class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.labels_df.columns = self.labels_df.columns.str.strip()
        self.root_dir = root_dir
        self.transform = transform

        self.label_column = [col for col in self.labels_df.columns if col != 'filename'][0]
        print(f"Detected label column: '{self.label_column}'")

        self.class_names = self.labels_df[self.label_column].unique()
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        print(f"Found classes in {os.path.basename(csv_file)}: {self.class_to_idx}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.loc[idx, 'filename'])
        image = Image.open(img_name).convert('RGB')

        # Use the automatically detected label column
        label_name = self.labels_df.loc[idx, self.label_column]
        label_idx = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)
        return image, label_idx

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': CustomCSVDataset(csv_file='train/_classes.csv', root_dir='train', transform=data_transforms['train']),
    'valid': CustomCSVDataset(csv_file='valid/_classes.csv', root_dir='valid', transform=data_transforms['valid'])
}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'valid']}
class_names = image_datasets['train'].class_names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[1].parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=10):
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}' + ' | ' + '-' * 10)
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print("Training complete.")
    return model

if __name__ == '__main__':
    trained_model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")