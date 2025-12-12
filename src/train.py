import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ResEmoteNet
from src.dataset import get_dataloaders
import time

def train():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DATA_DIR = "./data"  # Ensure this points to your extracted FER-2013 folder

    print(f"Initializing ResEmoteNet on {DEVICE}...")
    
    # Initialize Model, Loss, Optimizer
    model = ResEmoteNet(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load Data (Now includes your synthetic images!)
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.0f}s")

    # Save the trained model
    torch.save(model.state_dict(), "resemotenet.pth")
    print("Training Complete! Model saved as 'resemotenet.pth'")

if __name__ == "__main__":
    train()