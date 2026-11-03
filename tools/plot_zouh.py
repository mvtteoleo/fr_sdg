import numpy as np
import matplotlib.pyplot as plt


def get_data_plots():
    blue = np.genfromtxt("outputs/data_zouh_blue.csv", delimiter=",", usemask=False)

    data = np.genfromtxt("outputs/data_zouh.csv", delimiter=",", usemask=False)

    mask = (
        (data[:, 0] > 0.5)
        & (data[:, 1] <= 1.79)
        & (data[:, 1] >= 0.21)
        & (data[:, 0] <= 1.18)
    )

    def mask_fun(d):
        return (
            (d[:, 0] > 0.5) & (d[:, 1] <= 1.79) & (d[:, 1] >= 0.21) & (d[:, 0] <= 1.18)
        )

    fil = data[mask_fun(data)]

    blue = blue[mask_fun(blue)]
    return fil, blue


if __name__ == "__main__":
    fil, blue = get_data_plots()
    plt.plot(fil[:, 0], fil[:, 1])
    plt.plot(blue[:, 0], blue[:, 1])
    plt.show()
