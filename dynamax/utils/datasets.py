"""
Prepcocessing and data augmentation for the datasets.
"""
import os
import torchvision
import numpy as np
from augly import image
from typing import Tuple, Union
from multiprocessing import Pool
import jax.random as jr
import jax.numpy as jnp
from jax import vmap

class DataAugmentationFactory:
    """
    This is a base library to process / transform the elements of a numpy
    array according to a given function. To be used with gendist.TrainingConfig
    """
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, img, configs, n_processes=90):
        return self.process_multiple_multiprocessing(img, configs, n_processes)

    def process_single(self, X, *args, **kwargs):
        """
        Process a single element.

        Paramters
        ---------
        X: np.array
            A single numpy array
        kwargs: dict/params
            Processor's configuration parameters
        """
        return self.processor(X, *args, **kwargs)

    def process_multiple(self, X_batch, configurations):
        """
        Process all elements of a numpy array according to a list
        of configurations.
        Each image is processed according to a configuration.
        """
        X_out = []
                    
        for X, configuration in zip(X_batch, configurations):
            X_processed = self.process_single(X, **configuration)
            X_out.append(X_processed)
            
        X_out = np.stack(X_out, axis=0)
        return X_out
    
    def process_multiple_multiprocessing(self, X_dataset, configurations, n_processes):
        """
        Process elements in a numpy array in parallel.

        Parameters
        ----------
        X_dataset: array(N, ...)
            N elements of arbitrary shape
        configurations: list
            List of configurations to apply to each element. Each
            element is a dict to pass to the processor.
        n_processes: [int, None]
            Number of cores to use. If None, use all available cores.
        """
        num_elements = len(X_dataset)
        if type(configurations) == dict:
            configurations = [configurations] * num_elements

        if n_processes == 1:
            dataset_proc = self.process_multiple(X_dataset, configurations)
            return dataset_proc.reshape(num_elements, -1)

        dataset_proc = np.array_split(X_dataset, n_processes)
        config_split = np.array_split(configurations, n_processes)
        elements = zip(dataset_proc, config_split)

        with Pool(processes=n_processes) as pool:    
            dataset_proc = pool.starmap(self.process_multiple, elements)
            dataset_proc = np.concatenate(dataset_proc, axis=0)
        pool.join()

        return dataset_proc.reshape(num_elements, -1)


def load_mnist(root="./data", download=True):
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=download)
    images = np.array(mnist_train.data) / 255.0
    labels = mnist_train.targets

    mnist_test = torchvision.datasets.MNIST(root=root, train=False)
    images_test = np.array(mnist_test.data) / 255.0
    labels_test = mnist_test.targets

    train = (images, labels)
    test = (images_test, labels_test)
    return train, test


def rotate_mnist(X, angle):
    """
    Rotate an image by a given angle.
    We take the image to be a square of size 28x28.
    TODO: generalize to any size
    """
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    
    return X_shift

def generate_rotated_images(images, n_processes, minangle=0, maxangle=180):
    n_configs = len(images)
    processer = DataAugmentationFactory(rotate_mnist)
    angles = np.random.uniform(minangle, maxangle, n_configs)
    configs = [{"angle": float(angle)} for angle in angles]
    images_proc = processer(images, configs, n_processes=n_processes)
    return images_proc, angles


def load_rotated_mnist(
    root: str = "./data",
    target_digit: Union[int, None] = None,
    minangle: int = 0,
    maxangle: int = 180,
    n_processes: Union[int, None] = 1,
):
    """
    """
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 2)

    train, test = load_mnist(root=root)
    (X_train, labels_train), (X_test, labels_test) = train, test

    if target_digit is not None:
        X_train = X_train[labels_train == target_digit]
        X_test = X_test[labels_test == target_digit]
    
    n_train = len(X_train)
    X = np.concatenate([X_train, X_test], axis=0)
    (X, y) = generate_rotated_images(X, n_processes, minangle=minangle, maxangle=maxangle)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    train = (X_train, y_train)
    test = (X_test, y_test)
    return train, test


def load_1d_synthetic_dataset(n_train=100, n_test=100, key=0, sort_data=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key1, key2, subkey1, subkey2, key_shuffle = jr.split(key, 5)

    X_train = jr.uniform(key1, shape=(2*n_train, 1), minval=0.0, maxval=0.5)
    X_test = jr.uniform(key2, shape=(n_test, 1), minval=0.0, maxval=0.5)
    
    def generating_function(key, x):
        epsilons = jr.normal(key, shape=(3,))*0.02
        return (x + 0.3*jnp.sin(2*jnp.pi*(x+epsilons[0])) + 
                0.3*jnp.sin(4*jnp.pi*(x+epsilons[1])) + epsilons[2])
    
    keys_train = jr.split(subkey1, X_train.shape[0])
    keys_test = jr.split(subkey2, X_test.shape[0])
    y_train = vmap(generating_function)(keys_train, X_train)
    y_test = vmap(generating_function)(keys_test, X_test)

    # Standardize dataset
    X_train = (X_train - X_train.mean()) / X_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    y_test = (y_test - y_test.mean()) / y_test.std()

    sorted_idx = jnp.argsort(X_train.squeeze())
    train_idx = jnp.concatenate([
        sorted_idx[:n_train//2], sorted_idx[2*n_train - n_train//2:]
    ])

    X_train, y_train = X_train[train_idx], y_train[train_idx]

    if not sort_data:
        n_train = len(X_train)
        ixs = jr.choice(key_shuffle, shape=(n_train,), a=n_train)
        X_train = X_train[ixs]
        y_train = y_train[ixs]

    return (X_train, y_train), (X_test, y_test)
