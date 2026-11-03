from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

N = [x * 50 for x in range(1, 5)]
T = [1.28, 56.90, 146.4, 270.8]


def scale(x, A, B, C):
    return A + C * x**B


fit, _ = curve_fit(
    scale,
    N,
    T,
)

Ns = np.linspace(0, 200, num=5000)


plt.plot(Ns, scale(Ns, *fit), label="Fit")
plt.plot(N, T, "o")
plt.legend()
plt.show()
