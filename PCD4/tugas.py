import imageio.v3 as iio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


# ==========================================================
# 1. MEMBACA CITRA
# ==========================================================
image = iio.imread("images.jfif")


# ==========================================================
# 2. KONVERSI CITRA RGB MENJADI GRAYSCALE
# ==========================================================
if image.ndim == 3:
    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    ).astype(np.uint8)
else:
    gray = image.astype(np.uint8)


# ==========================================================
# 3. MENGHITUNG HISTOGRAM CITRA
# ==========================================================
histogram, bins = np.histogram(
    gray.flatten(),
    bins=256,
    range=(0, 256)
)


# ==========================================================
# 4. MENGHITUNG CUMULATIVE DISTRIBUTION FUNCTION (CDF)
# ==========================================================
cdf = histogram.cumsum()

# Mengambil nilai CDF minimum yang bukan nol
cdf_min = cdf[cdf > 0][0]


# ==========================================================
# 5. NORMALISASI CDF
# ==========================================================
cdf_normalized = (
    (cdf - cdf_min) /
    (gray.size - cdf_min)
) * 255

cdf_normalized = np.clip(
    cdf_normalized,
    0,
    255
).astype(np.uint8)


# ==========================================================
# 6. MELAKUKAN HISTOGRAM EQUALIZATION
# ==========================================================
equalized = cdf_normalized[gray]


# ==========================================================
# 7. MENGGUNAKAN SCIPY.NDIMAGE
#    Gaussian Filter untuk proses smoothing
# ==========================================================
smoothed = ndimage.gaussian_filter(
    equalized.astype(np.float32),
    sigma=1
)

smoothed = np.clip(
    smoothed,
    0,
    255
).astype(np.uint8)


# ==========================================================
# 8. MENYIMPAN HASIL
# ==========================================================
iio.imwrite("hasil_grayscale.png", gray)
iio.imwrite("hasil_equalisasi.png", equalized)
iio.imwrite("hasil_smoothing.png", smoothed)


# ==========================================================
# 9. MENAMPILKAN HASIL
# ==========================================================
plt.figure(figsize=(15, 9))


# ----------------------------------------------------------
# Citra Awal
# ----------------------------------------------------------
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Citra Awal", fontsize=12)
plt.axis("off")


# ----------------------------------------------------------
# Citra Grayscale
# ----------------------------------------------------------
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Grayscale", fontsize=12)
plt.axis("off")


# ----------------------------------------------------------
# Hasil Histogram Equalization
# ----------------------------------------------------------
plt.subplot(2, 3, 3)
plt.imshow(equalized, cmap="gray", vmin=0, vmax=255)
plt.title("Hasil Histogram Equalization", fontsize=12)
plt.axis("off")


# ----------------------------------------------------------
# Histogram Sebelum Equalisasi
# ----------------------------------------------------------
plt.subplot(2, 3, 4)
plt.hist(
    gray.flatten(),
    bins=256,
    range=(0, 256)
)
plt.title("Histogram Sebelum Equalisasi", fontsize=12)
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)


# ----------------------------------------------------------
# Histogram Setelah Equalisasi
# ----------------------------------------------------------
plt.subplot(2, 3, 5)
plt.hist(
    equalized.flatten(),
    bins=256,
    range=(0, 256)
)
plt.title("Histogram Setelah Equalisasi", fontsize=12)
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)


# ----------------------------------------------------------
# Hasil Gaussian Filter
# ----------------------------------------------------------
plt.subplot(2, 3, 6)
plt.imshow(smoothed, cmap="gray", vmin=0, vmax=255)
plt.title("Hasil Gaussian Filter", fontsize=12)
plt.axis("off")


# ==========================================================
# 10. MERAPIKAN TAMPILAN
# ==========================================================
plt.suptitle(
    "Pengolahan Citra dan Histogram Equalization",
    fontsize=16
)

plt.tight_layout()
plt.show()