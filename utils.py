import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_metrics(metrics, num_bootstraps):
    times = range(len(metrics["loss_b1"]))
    
    # 1. Plot for loss across different bootstraps
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    for i in range(1, num_bootstraps + 1):
        plt.plot(times, metrics[f"loss_b{i}"], label=f'Bootstrap {i}')
    plt.title('Loss per Bootstrap')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Plot for gradient norms across different bootstraps
    plt.subplot(1, 3, 2)
    for i in range(1, num_bootstraps + 1):
        plt.plot(times, metrics[f"grad_norms_b{i}"], label=f'Gradient Norms Bootstrap {i}')
    plt.title('Gradient Norms per Bootstrap')
    plt.xlabel('Time')
    plt.ylabel('Gradient Norm')
    plt.legend()
    
    # 3. Plot for intrinsic reward for different digit prediction types
    plt.subplot(1, 3, 3)
    plt.plot(times, metrics["ir_0"], label='Intrinsic Reward for 0s')
    plt.plot(times, metrics["ir_1"], label='Intrinsic Reward for 1s')
    plt.title('Intrinsic Reward per Digit Type')
    plt.xlabel('Time')
    plt.ylabel('Intrinsic Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.show()

def plot_predictions(state, batch):
    """
    Plots side-by-side comparisons of original digits and their predicted transformations.

    Args:
        state (train_state.TrainState): The training state containing the model parameters.
        batch (jnp.array): A batch of input, target pairs of shape (B, 2, 28, 28)
    """
    inputs = batch[:, 0, :, :]
    targets = batch[:, 1, :, :]
    num_samples = batch.shape[0]

    # Generate predictions
    predictions = state.apply_fn({'params': state.params}, inputs)

    fig, axs = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    for i in range(num_samples):
        # Display input images
        axs[0, i].imshow(inputs[i].reshape(28, 28), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('Original')

        # Display target images
        axs[1, i].imshow(targets[i].reshape(28, 28), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title('Target')

        # Display predictions
        axs[2, i].imshow(predictions[i].reshape(28, 28), cmap='gray')
        axs[2, i].axis('off')
        axs[2, i].set_title('Prediction')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
