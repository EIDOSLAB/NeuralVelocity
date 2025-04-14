import os
import random

import numpy as np
import torch
import torchvision.transforms as T


# Defining a method to add a patch to an image array
def _get_image(self, img, color):
    # Getting the patch coordinates and color
    x1, y1 = self.get_xy()  # Getting the lower left corner of the square
    x2, y2 = x1 + self.get_width(), y1 + self.get_height()  # Getting the upper right corner of the square

    # Getting the rotation matrix
    theta = self.get_angle() * np.pi / 180
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Looping over the pixels in the image array
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Computing the pixel coordinates relative to the patch center
            x = i - (x1 + x2) / 2
            y = j - (y1 + y2) / 2

            # Rotating the pixel coordinates
            x, y = R @ np.array([x, y])

            # Checking if the pixel is inside the patch
            if -self.get_width() / 2 <= x <= self.get_width() / 2 and -self.get_height() / 2 <= y <= self.get_height() / 2:
                # Computing the distance from the pixel to the center of the patch
                d = np.sqrt(x ** 2 + y ** 2)

                a = -100
                b = 100
                # Applying a sigmoid function to create a leaf-like shape
                z = 1 / (1 + np.exp(-a * (d - b)))  # Changing the values of a and b

                # Applying a sigmoid function to create a leaf-like shape
                # z = 1 / (1 + np.exp(-10 * (d - self.get_width() / 4)))

                # Updating the pixel color with a weighted average of the patch color and the background color
                img[i, j] = z * np.array(color[:3]) + (1 - z) * img[
                    i, j]  # Converting the color array to a numpy array

    # Returning the updated image array
    return img


def generate_random_dataset(img_shape: tuple = (3, 32, 32),
                            dataset_size: int = 10,
                            save_to_disk: bool = True,
                            output_folder_path: str = None,
                            seed: int = 0,
                            apply_transforms=True):
    """

    """
    # if save_to_disk is true we must give an output path
    assert save_to_disk and output_folder_path or not save_to_disk
    # make sure that the output path exists
    if save_to_disk and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # init seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    transform = T.ToPILImage() if apply_transforms else None
    images = []

    n_zeros = len(str(dataset_size))
    # Generate images
    for idx in range(dataset_size):
        print(f"Status: {idx + 1}/{dataset_size}")
        img = torch.randn(*img_shape)
        if transform:
            img = transform(img)
        if save_to_disk:
            img.save(os.path.join(output_folder_path, f"{str(idx).zfill(n_zeros)}.jpg"))
        else:
            images.append(img)
    return None if save_to_disk else images


def generate_dead_leaves_dataset(img_shape=256,
                                 dataset_size: int = 10,
                                 save_to_disk: bool = True,
                                 output_folder_path: str = None,
                                 seed=0,
                                 n_leaves=16,
                                 n_colors=16,
                                 specific_cmap=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Generate random image data
    np.random.seed(seed)
    random.seed(seed)

    # Assigning the method to the patch object
    patches.Rectangle.get_image = _get_image
    # Defining parameters
    n = img_shape  # Image size
    m = n_leaves  # Number of leaves per row and column
    k = n_colors  # Number of colors

    n_zeros = len(str(dataset_size))
    images = []
    for idx in range(dataset_size):
        print(f"Processing image: {idx + 1}/{dataset_size}")
        if specific_cmap:
            cmap = plt.cm.get_cmap(specific_cmap, k)  # Color map
        else:
            cmap = plt.cm.get_cmap(random.choice(['Spectral', 'Dark2', 'rainbow', 'turbo', 'tab20']), k)

        # Creating an empty image array
        img = np.zeros((n, n, 3))

        # Looping over the leaves
        for i in range(m):
            for j in range(m):
                # Computing the center and size of each leaf with some random noise
                x = (i + 0.5 + np.random.uniform(-2., 2.)) * n / m  # Adding some noise to the x coordinate
                y = (j + 0.5 + np.random.uniform(-2., 2.)) * n / m  # Adding some noise to the y coordinate
                s = (0.5 + np.random.uniform(-5, 5)) * n / m  # Adding some noise to the size
                # Computing the angle of rotation based on a wavelet function
                theta = np.pi * np.sin(2 * np.pi * (x + y) / n)

                # Creating a patch object for each leaf using a square shape
                patch = patches.Rectangle((x - s / 2, y - s / 2), s, s, angle=theta * 180 / np.pi)

                # Assigning a random color to each leaf from the color map
                color = cmap(np.random.randint(k))

                # Adding the patch to the image array
                img = patch.get_image(img, color)
        plt.imshow(img)
        plt.axis('off')
        # img = transform(img_color)
        if save_to_disk:
            plt.savefig(os.path.join(output_folder_path, f"{str(idx).zfill(n_zeros)}.jpg"), bbox_inches='tight',
                        transparent=True, pad_inches=0)
        else:
            images.append(img)

    return None if save_to_disk else images


def generate_wavelet_dataset(img_shape=256, color_spectrum="Blues", color=0,
                             seed=0,
                             dataset_size: int = 10,
                             save_to_disk: bool = True,
                             output_folder_path: str = None):
    import matplotlib.pyplot as plt
    import pywt
    color_spectrum = plt.cm.get_cmap(random.choice(['Spectral', 'Dark2', 'rainbow', 'turbo', 'tab20']), 8)
    # Generate random image data
    random.seed(seed)
    np.random.seed(seed)

    full_img_shape = (img_shape, img_shape, 3)
    n_zeros = len(str(dataset_size))
    images = []
    for idx in range(dataset_size):
        print(f"Processing image: {idx + 1}/{dataset_size}")
        data = np.random.wald(4, 2, full_img_shape)

        # Apply wavelet transform
        coeffs = pywt.dwt2(data[:, :, 0], 'haar')
        cA, (cH, cV, cD) = coeffs

        # Modify coefficients based on distribution
        # if distribution == 'WMM':
        cH *= 0.25
        cV *= 0.25
        cD *= 0.25

        # Apply inverse wavelet transform
        coeffs = cA, (cH, cV, cD)
        reconstructed_data = pywt.idwt2(coeffs, 'haar')

        # Normalize data to [0, 1]
        normalized_data = (reconstructed_data - np.min(reconstructed_data)) / (
                np.max(reconstructed_data) - np.min(reconstructed_data))

        # Apply color spectrum and color to image
        colored_data = np.zeros(full_img_shape)
        for i in range(3):
            colored_data[:, :, i] = normalized_data * color_spectrum[i]
        colored_data += color
        plt.imshow(colored_data)
        plt.axis('off')
        if save_to_disk:
            plt.savefig(os.path.join(output_folder_path, f"{str(idx).zfill(n_zeros)}.jpg"), bbox_inches='tight',
                        transparent=True, pad_inches=0)
        else:
            images.append(colored_data)

    return None if save_to_disk else images
