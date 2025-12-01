import numpy as np
import pandas as pd
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate 100 samples of 3D coordinates (x, y, z) in range 0-10
num_samples = 100
x = np.random.uniform(0, 10, num_samples)
y = np.random.uniform(0, 10, num_samples)
z = np.random.uniform(0, 10, num_samples)

# Target: distance from origin (spatial function)
target = np.sqrt(x**2 + y**2 + z**2)

# Create DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'target': target
})

# Save to CSV
df.to_csv('test_3d_dataset.csv', index=False)

print(f"Generated 3D dataset with {num_samples} samples")
print("Features: x, y, z (coordinates)")
print("Target: distance from origin")
print("Saved to test_3d_dataset.csv")
print("\nFirst few rows:")
print(df.head())