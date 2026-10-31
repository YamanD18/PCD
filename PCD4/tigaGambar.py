from imageio import v3 as image
import matplotlib.pyplot as plt
import numpy as np

# Lokasi gambar
path = "images.jfif"

# Membaca gambar
my_image = image.imread(path)

# Membuat gambar lebih terang
bright_image = np.clip(my_image * 1.5, 0, 255).astype(np.uint8)

# Membuat gambar lebih gelap
dark_image = (my_image * 0.5).astype(np.uint8)

# Menampilkan 3 gambar
plt.figure(figsize=(15, 5))

# Gambar asli
plt.subplot(1, 3, 1)
plt.imshow(my_image)
plt.title("Gambar Asli")
plt.axis("off")

# Gambar terang
plt.subplot(1, 3, 2)
plt.imshow(bright_image)
plt.title("Gambar Terang")
plt.axis("off")

# Gambar gelap
plt.subplot(1, 3, 3)
plt.imshow(dark_image)
plt.title("Gambar Gelap")
plt.axis("off")

plt.show()