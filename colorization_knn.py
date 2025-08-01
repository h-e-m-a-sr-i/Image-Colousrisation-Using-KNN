import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load sample color image (for training)
train_img = cv2.imread('grayscale/sample_input.jpg')
train_img = cv2.resize(train_img, (100, 100))  # Resize to reduce computation

# Convert to LAB color space
lab = cv2.cvtColor(train_img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Prepare training data
X_train = L.flatten().reshape(-1, 1)
y_train = np.stack((A.flatten(), B.flatten()), axis=1)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Load grayscale image to colorize
gray_img = cv2.imread('grayscale/sample_input.jpg', cv2.IMREAD_GRAYSCALE)
gray_img = cv2.resize(gray_img, (100, 100))  # Match training image size

# Predict A and B channels
X_test = gray_img.flatten().reshape(-1, 1)
predicted_ab = knn.predict(X_test)

# Reconstruct LAB image
A_pred = predicted_ab[:, 0].reshape(gray_img.shape)
B_pred = predicted_ab[:, 1].reshape(gray_img.shape)

colorized_lab = cv2.merge([gray_img, A_pred, B_pred]).astype(np.uint8)
colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

# Save output
cv2.imwrite('output/colorized_output.jpg', colorized_bgr)

print("Colorized image saved to output/colorized_output.jpg")
