import numpy as np


# ref: Competitive Code
def GaborFilter_cc(len_filter, sigma=4.6, delta=2.6, num_direction=6):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_len = int(len_filter / 2)

    Filter = np.zeros((num_direction, len_filter, len_filter))
    for a in range(num_direction):
        theta = np.pi / 2 - np.pi * a / num_direction  # direction angle
        kappa = np.sqrt(2 * np.log(2)) * (delta + 1) / (delta - 1)
        w = kappa / sigma
        fFactor1 = -w / (np.sqrt(2 * np.pi) * kappa)
        fFactor2 = -(w * w) / (8 * kappa * kappa)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for c in range(len_filter):  # col
            x = c - half_len
            for r in range(len_filter):  # row
                y = r - half_len
                x1 = x * cos_theta + y * sin_theta
                y1 = y * cos_theta - x * sin_theta
                f_comp = fFactor1 * np.exp(fFactor2 * (4 * x1 * x1 + y1 * y1))
                Filter[a, r, c] = f_comp * np.cos(w * x1)  # same with palm lines

        Filter[a, :, :] -= Filter[a, :, :].mean()

    return Filter


# ref http://en.wikipedia.org/wiki/Gabor_filter
def GaborFilter(ksize, num_direction, sigma, lambd, gamma):
    assert num_direction % 2 == 0, 'num_direction should be an even number!'
    half_size = int(ksize/2)
    sigma2 = 2*sigma**2

    Filter = np.zeros((num_direction, ksize, ksize))
    l_min = -half_size
    l_max = half_size
    x, y = np.meshgrid(range(l_min, l_max + 1), range(l_min, l_max + 1))
    for a in range(num_direction):
        theta = np.pi * a / num_direction  # direction angle
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_theta = x * cos_theta + y * sin_theta
        y_theta = y * cos_theta - x * sin_theta
        term1 = np.exp(-(x_theta ** 2 + (gamma * y_theta) ** 2) / sigma2)
        term2 = np.cos(2 * np.pi * x_theta / lambd)
        Filter[a] = term1 * term2
        Filter[a] -= Filter[a].mean()

    return Filter


def GaborArray(sigma=4.85, wavelength=14.1, ratio=1.92):
    halfLength = 17

    xmax = halfLength
    xmin = -halfLength
    ymax = halfLength
    ymin = -halfLength
    [x, y] = np.meshgrid(range(xmin, xmax+1), range(ymin, ymax+1))

    mask = np.ones((35, 35))
    for row in range(1,36):
        for col in range(1,36):
            if (row - 18)**2 + (col - 18)**2 > 289:
                mask[row-1, col-1] = 0

    gb_r = np.zeros((6, 35, 35))
    for oriIndex in range(1, 7):
        theta = np.pi / 6 * (oriIndex - 1)

        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5*(x_theta**2/sigma**2 + y_theta**2/(ratio*sigma)**2)) * np.cos(2*np.pi/wavelength*x_theta)

        total = (gb * mask).sum()
        meanInner = total / (mask).sum()

        gb = gb - meanInner.mean()
        gb = gb * mask
        gb_r[oriIndex-1] = gb

    return gb_r


if __name__ == '__main__':
    len_filter = 35
    sigma = 4.6  # line width frequency
    delta = 2.6  # line length frequency
    num_direction = 6
    g1 = GaborFilter_cc(len_filter, sigma, delta, num_direction)
    g2 = GaborFilter(len_filter, num_direction, lambd=14.1, sigma=4.85, gamma=1/1.92)
    plt.imshow(g1[0, :, :], cmap=plt.cm.gray_r)
    plt.show()
    plt.imshow(g2[0, :, :], cmap=plt.cm.gray_r)
    plt.show()
