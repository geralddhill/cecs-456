import numpy as np
import scipy
import matplotlib.pyplot as plt


def main():
    # Setup
    rng = np.random.default_rng(42)
    N = 300
    dimensions = [1, 5, 10, 50, 100, 1000, 5000, 10000]
    distances = []

    # Generate random data
    for d in dimensions:
        sample = rng.random((N, d), dtype=np.float32)
        data = scipy.spatial.distance.pdist(sample, metric='euclidean')
        distances.append(np.mean(data))

    # Print Table
    if len(dimensions) != len(distances):
        raise ValueError('The number of dimensions does not match the number of samples')
    print("d         mean_distance")
    for i in range(len(distances)):
        print(f"{dimensions[i]:<10}{distances[i]}")

    # Plot
    fig, ax = plt.subplots()
    ax.semilogx(dimensions, distances, "o--")
    ax.set(title='Average Pairwise Distance in High Dimensions',
           xticks=dimensions, xticklabels=dimensions,
           xlabel='Dimensions', ylabel='Distance')
    ax.grid()
    ax.grid(which="minor", color="0.9")
    plt.show()


if __name__ == '__main__':
    main()