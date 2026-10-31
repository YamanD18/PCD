from imageio import v3 as image
import matplotlib.pyplot as plt
import numpy as np

# Lokasi gambar
path = "images.jfif"

# Baca gambar
my_image = image.imread(path)

# Perbesar kegelapan
dark_image = (my_image * 0.3).astype(np.uint8)

# Tampilkan gambar asli dan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(my_image)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dark_image)
plt.title("Gambar Lebih Gelap")
plt.axis("off")

plt.show()