import matplotlib.pyplot as plt
from data import MNISTOneStep

def visualize_pairs(dataset):
    for i in range(len(dataset)):
        pair = dataset[i]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Display the first image
        axes[0].imshow(pair[0], cmap='gray')
        axes[0].set_title("First Image")
        axes[0].axis('off')
        
        # Display the paired image
        axes[1].imshow(pair[1], cmap='gray')
        axes[1].set_title("Paired Image")
        axes[1].axis('off')
        
        plt.show()

# Load the dataset
mnist_onestep = MNISTOneStep(shuffle=True)

# Visualize the pairs
visualize_pairs(mnist_onestep)
