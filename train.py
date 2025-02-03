import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvNet
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensures all images are grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root="/Users/sarayu.madakasira/CNN/mnist_png/training", transform=transform)
test_data = datasets.ImageFolder(root="/Users/sarayu.madakasira/CNN/mnist_png/testing", transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

image, label = train_data[0]

print(image.size())

model = ConvNet()
loss_f = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(model, train_loader, criterion, optimizer, epochs=5):
    model.apply(init_weights)
    model.train()  

    losses = [] 
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item()
            losses.append(loss.item())
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
    plt.plot(losses, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs Iterations")
    plt.legend()
    plt.show()

train(model, train_loader, loss_f, optimizer, epochs=5)
torch.save(model.state_dict(), "model.pth")
print("Model saved successfully as 'model.pth'")
