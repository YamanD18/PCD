import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('view.png')
if img is None:
    raise FileNotFoundError("kunci.jpeg tidak ditemukan")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharp = cv2.filter2D(rgb, -1, kernel)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, im, title in zip(ax, [rgb, sharp], ['Asli', 'Sharpening']):
    a.imshow(im)
    a.set_title(title)
    a.axis('off')

plt.tight_layout()
plt.show()