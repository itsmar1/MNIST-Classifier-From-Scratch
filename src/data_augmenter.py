import numpy as np


class DataAugmenter:
    """Data augmentation techniques for MNIST"""

    @staticmethod
    def shift_image(image, dx, dy):
        """Shift image by dx and dy pixels"""
        image = image.reshape(28, 28)
        shifted = np.roll(image, dx, axis=1)
        shifted = np.roll(shifted, dy, axis=0)
        # Zero out rolled edges
        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0
        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0
        return shifted.reshape(-1)

    @staticmethod
    def add_noise(image, noise_factor=0.05):
        """Add random noise to image"""
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)

    @staticmethod
    def rotate_image(image, angle):
        """Simple rotation approximation using shear"""
        from scipy.ndimage import rotate
        image_2d = image.reshape(28, 28)
        rotated = rotate(image_2d, angle, reshape=False, order=1)
        return rotated.reshape(-1)

    @staticmethod
    def augment_batch(X_batch, y_batch, augmentation_factor=2):
        """Augment a batch of images"""
        augmented_X = []
        augmented_y = []

        for i in range(len(X_batch)):
            image = X_batch[i]
            label = y_batch[i]

            # Original image
            augmented_X.append(image)
            augmented_y.append(label)

            # Apply augmentations
            for _ in range(augmentation_factor - 1):
                aug_image = image.copy()

                # Randomly choose augmentation
                aug_type = np.random.choice(['shift', 'noise', 'rotate'])

                if aug_type == 'shift':
                    dx = np.random.randint(-3, 4)
                    dy = np.random.randint(-3, 4)
                    aug_image = DataAugmenter.shift_image(image, dx, dy)
                elif aug_type == 'noise':
                    aug_image = DataAugmenter.add_noise(image, 0.1)
                else:  # rotate
                    angle = np.random.randint(-15, 16)
                    aug_image = DataAugmenter.rotate_image(image, angle)

                augmented_X.append(aug_image)
                augmented_y.append(label)

        return np.array(augmented_X), np.array(augmented_y)