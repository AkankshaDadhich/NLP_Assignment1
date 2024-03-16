import matplotlib.pyplot as plt
import numpy as np

def show_heat_map(weights):
    weights = np.array(weights)
    #weights = abs(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            plt.text(j, i, f'{weights[i, j]:.2f}', ha='center', va='center', color='white')

    plt.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar()  # Add colorbar for reference
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    plt.title("Dense Layer Weights Heatmap")
    plt.show()
