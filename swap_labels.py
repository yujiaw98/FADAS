import torch
from torchvision import datasets, transforms

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

def swap_labels(label):
    # Swap each label i with i+1, and 9 with 0
    return (label + 1) % 10

# Apply label swapping
for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    train_dataset.targets[i] = swap_labels(label)

for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    test_dataset.targets[i] = swap_labels(label)

# Now train_dataset and test_dataset have swapped labels
