import torch
from model import ConvNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

model = ConvNet()
model.load_state_dict(torch.load('/Users/sarayu.madakasira/CNN/model.pth', map_location=torch.device('cpu')))  # Load weights from saved model file
model.eval()  

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure it's a 1-channel (grayscale) image
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root="/Users/sarayu.madakasira/CNN/mnist_png/testing", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def make_inference(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  
        for images, labels in test_loader:
            outputs = model(images)  
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()
    
    accuracy = 100*correct/total
    print(f'Accuracy on test set: {accuracy}%')

make_inference(model, test_loader)


def make_inferences(model, test_loader):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():  
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)   
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    f1 = f1_score(all_labels, all_predictions, average=None)
    print(f'F1 Scores for each class: {f1}')

make_inferences(model, test_loader)

make_inference(model, test_loader)
