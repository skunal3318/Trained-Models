import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(img, noise_type="gaussian", sigma=25):
    """Add Gaussian or Salt & Pepper noise"""
    if noise_type == "gaussian":
        gauss = np.random.normal(0, sigma, img.shape)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = img.copy()
        # Salt
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords[0], coords[1]] = 0
        return out

def create_low_pass_filter(shape, cutoff_ratio=0.1):
    """Create circular low-pass filter mask"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    # Circular mask
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    
    mask[radius < cutoff_ratio * min(crow, ccol)] = 1
    return mask

def fourier_denoise(image_path, cutoff_ratio=0.1, noise_type="gaussian", sigma=25):
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found!")
    
    # 2. Add noise
    noisy = add_noise(img, noise_type=noise_type, sigma=sigma)

    # 3. FFT (use float32)
    f = np.fft.fft2(noisy.astype(np.float32))
    fshift = np.fft.fftshift(f)  # Center the low frequencies

    # 4. Magnitude spectrum (for visualization)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 5. Create low-pass filter
    mask = create_low_pass_filter(noisy.shape, cutoff_ratio=cutoff_ratio)

    # 6. Apply filter
    fshift_filtered = fshift * mask

    # 7. Inverse shift and inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_denoised = np.abs(img_back)
    img_denoised = np.clip(img_denoised, 0, 255).astype(np.uint8)

    # 8. Plot everything
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f"Noisy Image ({noise_type})")
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("FFT Magnitude Spectrum")
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Low-Pass Filter Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Filtered Spectrum")
    plt.imshow(20 * np.log(np.abs(fshift_filtered) + 1), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Denoised Image")
    plt.imshow(img_denoised, cmap='gray')
    plt.axis('off')

    plt.suptitle(f"Image Denoising via Fourier Transform (Cutoff: {cutoff_ratio})", fontsize=16)
    plt.tight_layout()
    plt.show()

    return img_denoised

# ——— RUN THE PROJECT ———
if __name__ == "__main__":
    # Replace with your image path
    image_path = "brain_mri.jpg"  # Try a medical image or any grayscale photo
    denoised = fourier_denoise(
        image_path=image_path,
        cutoff_ratio=0.15,      # Try 0.05 (strong filter) to 0.3 (weak)
        noise_type="gaussian",  # or "s&p"
        sigma=30
    )
