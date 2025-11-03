import numpy as np
import matplotlib.pyplot as plt

# Hinge function: min(0, ν + c)
def hinge(nu, c=0.0):
    return np.minimum(0.0, nu + c)

# Range for ν and chosen offset c
nu = np.linspace(-3.0, 3.0, 400)
c = 0.0  # change this to visualize different offsets
h = hinge(nu, c)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(nu, h, label=f"min(0, ν + {c})", linewidth=2)
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(-c, color="gray", linewidth=0.8, linestyle=":")
plt.legend()
plt.title("Hinge function: min(0, ν + c)")
plt.xlabel("ν")
plt.ylabel("value")
plt.grid(True)
plt.tight_layout()
plt.show()
