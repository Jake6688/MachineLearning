import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a random 3x3 matrix
matrix = np.random.rand(3, 3)
matrix = matrix @ matrix.T  # Make the matrix symmetric
print(f"Matrix:\n{matrix}\n")

# Compute the determinant
det = np.linalg.det(matrix)
print(f"Determinant: {det}\n")

# Compute the inverse
if det != 0:
    inverse = np.linalg.inv(matrix)
    print(f"Inverse:\n{inverse}\n")
else:
    print("Matrix is singular and cannot be inverted.\n")

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(f"Eigenvalues: {eigenvalues}\n")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Verify the result of matrix multiplication
product = matrix @ inverse if det != 0 else None
print(f"Product of matrix and its inverse (should be identity matrix):\n{product}\n")

# Plot the original matrix
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Matrix")
plt.imshow(matrix, cmap='viridis', interpolation='none')
plt.colorbar()

# Plot the inverse matrix if it exists
if det != 0:
    plt.subplot(1, 3, 2)
    plt.title("Inverse Matrix")
    plt.imshow(inverse, cmap='viridis', interpolation='none')
    plt.colorbar()

# Plot the eigenvectors in 3D
ax = plt.subplot(1, 3, 3, projection='3d')
ax.set_title("Eigenvectors")
origin = np.zeros((3, 3))  # origin point
ax.quiver(origin[0], origin[1], origin[2], eigenvectors[0], eigenvectors[1], eigenvectors[2])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid()

plt.tight_layout()
plt.show()