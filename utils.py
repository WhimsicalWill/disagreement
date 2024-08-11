import matplotlib.pyplot as plt

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
    plt.show()

