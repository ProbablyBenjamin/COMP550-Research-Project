from dataset.get_dataset import get_preprocessed_instances, get_labels
import random
import numpy as np

def random_split(x, y, seed_set_size):
    indices = random.sample(range(len(x)), seed_set_size)
    seed_x = [x[i] for i in indices]
    seed_y = [y[i] for i in indices]
    return seed_x, seed_y

def uniform_split(x, y, per_class_size):
    x = np.array(x)
    y = np.array([int(l) for l in y])
    unique_labels = np.unique(y, axis = 0)
    seed_x, seed_y = [], []
    for label in unique_labels:
        indices = np.argwhere(y == label).flatten()
        random_samples = np.random.choice(indices, per_class_size, replace=False)
        seed_x.append(x[random_samples])
        seed_y.append(y[random_samples])
    seed_x = np.concatenate(seed_x)
    seed_y = np.concatenate(seed_y)
    return seed_x, [str(y) for y in seed_y] #GAURAV why this no return striiiinggg :(

if __name__ == "__main__":
    x, y = get_preprocessed_instances(), get_labels()
    seed_x, seed_y = uniform_split(x, y, 1)
    for x, y in zip(seed_x, seed_y):
        print (x, y)
