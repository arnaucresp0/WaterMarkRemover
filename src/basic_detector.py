import cv2
import numpy as np
import os

# Carrega la imatge
img_path = os.path.join("..", "images", "test_image.jpg")
image = cv2.imread(img_path)

if image is None:
    print(f"No s'ha pogut carregar la imatge: {img_path}")
    exit(1)

# Escala de grisos
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filtrat i binarització per intentar ressaltar la marca d'aigua
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

# Detecció de contorns
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibuixa els contorns trobats sobre la imatge original
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Mostra les imatges
cv2.imshow("Original", image)
cv2.imshow("Grayscale + Threshold", thresh)
cv2.imshow("Contorns detectats", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
