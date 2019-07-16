import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial(polynomial):
    freq = np.linspace(- np.pi, np.pi, 1000)
    plane = [np.exp(1j * f) for f in freq]
    response = [1.0 / np.polyval(polynomial, z) for z in plane]
    plt.plot(freq, response)
    plt.xlim(- np.pi, np.pi)
    plt.ylim(-40, 40)
    plt.show()

def plot_poles(poles):
    #range = max(max(abs(poles.real)), max(abs(poles.imag)))
    t = np.linspace(0,2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    plt.scatter([x.real for x in poles],[x.imag for x in poles], color='red')
    plt.axes().set_aspect('equal')
    #plt.xlim(-1.1 * range,1.1 * range)
    #plt.ylim(-1.1 * range,1.1 * range)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()

def plot_fourier(fourier):
    norm = [np.absolute(bin) for bin in fourier]
    order = range(len(fourier))
    plt.scatter(order, norm, color = 'r')
    plt.plot([0, len(fourier) - 1], [1, 1], color = 'b', linestyle = '-', linewidth = 1)
    plt.xlim(0, len(fourier) - 1)
    plt.ylim(0, 10)
    plt.show()

def poles_from_polynomial(polynomial):
    poles = np.roots(polynomial)
    return poles

def polynomial_from_poles(poles):
    polynomoial = np.poly(poles)
    return polynomoial

def coeffs_to_polynomial(coeffs):
    polynomial = [1 for _ in range(len(coeffs) + 1)]

    for i in range(1, len(coeffs) + 1):
        polynomial[i] = - coeffs[len(coeffs) - i]
    return polynomial

def coeffs_from_polynomial(polynomial):
    coeffs = [0 for _ in range(len(polynomial) - 1)]

    for i in range(len(polynomial) - 1):
        coeffs[i] = - polynomial[len(polynomial) - 1 - i]
    return coeffs

def coeffs_to_fourier(coeffs):
    fourier = [0 + 0j for _ in range(len(coeffs))]
    const = -2.0j * np.pi / len(coeffs)

    for k in range(len(coeffs)):
        for n in range(len(coeffs)):
            fourier[k] += coeffs[n] * np.exp(k * n * const)
    return fourier

def coeffs_to_fourier_slanted(coeffs, slant):
    fourier = [0 + 0j for _ in range(len(coeffs))]
    const = -2.0j * np.pi / len(coeffs)

    for k in range(len(coeffs)):
        for n in range(len(coeffs)):
            fourier[k] += coeffs[n] * np.exp((k + slant) * n * const)
    return fourier

def coeffs_from_fourier(fourier):
    coeffs = [0 for _ in range(len(fourier))]
    const = 2.0j * np.pi / len(fourier)

    for n in range(len(coeffs)):
        for k in range(len(coeffs)):
            coeffs[n] += fourier[k] * np.exp(k * n * const) / len(fourier)
    return coeffs

def coeffs_from_fourier_slanted(fourier, slant):
    coeffs = [0 for _ in range(len(fourier))]
    const = 2.0j * np.pi / len(fourier)

    for n in range(len(coeffs)):
        for k in range(len(coeffs)):
            coeffs[n] += fourier[k] * np.exp((k + slant) * n * const) / len(fourier)
    return coeffs

def clamp_fourier(fourier, epsilon):
    result = [0 for _ in range(len(fourier))]

    for i in range(len(fourier)):
        norm = np.absolute(fourier[i])
        if norm > 1.0 - epsilon:
            result[i] = fourier[i] / norm * (1.0 - epsilon)
        else:
            result[i] = fourier[i]
    return result

"""
num = 10
for freq in np.linspace(0, num / 2, 20):
    coeffs = [0 for _ in range(num)]
    for i in range(num):
        coeffs[i] = np.cos((i * 1.0 / num - 1.0) * 2.0 * np.pi * freq) / num

    fourier = coeffs_to_fourier(coeffs)
    first = 0
    last = 0;
    for bin in fourier:
        first += 1.0 / num * bin
        last += 1.0 / num * bin * bin

    print("===================")
    print("first\t: %f" % (np.absolute(first)))
    print("last\t: %f" % (np.absolute(last)))
    print("stable\t: %r" % (np.absolute(first) >= np.absolute(last)))

    polynomial = coeffs_to_polynomial(coeffs)
    poles = poles_from_polynomial(polynomial)

    plot_polynomial(polynomial)
    #plot_poles(poles)
"""

"""
num = 20

for i in range(1000):
    coeffs = np.random.normal(0, 0.2, num)
    polynomial = coeffs_to_polynomial(coeffs)
    poles = poles_from_polynomial(polynomial)
    fourier = coeffs_to_fourier(coeffs)

    index = 1
    maximum = coeffs[1]
    for i in range(2, num):
        if coeffs[i] >

    ground = True
    for pole in poles:
        if np.absolute(pole) > 1.0:
            ground = False

    print("===================")
    print("first\t: %f" % (np.absolute(first)))
    print("last\t: %f" % (np.absolute(last)))
    print("exp\t\t: %r" % (exp))
    print("ground\t: %r" % (ground))

    if (not exp) and ground:
        print("bad news")
"""

"""
coeffs = np.random.normal(0, 1.0, 20)
polynomial = coeffs_to_polynomial(coeffs)
poles = poles_from_polynomial(polynomial)
fourier = coeffs_to_fourier(coeffs)

plot_poles(poles)
plot_fourier(fourier)

print("======= before fourier clamp =======")
for pole in poles:
    print(np.absolute(pole))

fourier = clamp_fourier(fourier, 0.01)
coeffs = coeffs_from_fourier(fourier)
polynomial = coeffs_to_polynomial(coeffs)
poles = poles_from_polynomial(polynomial)

plot_poles(poles)
plot_fourier(fourier)

print("======= after fourier clamp =======")
for pole in poles:
    print(np.absolute(pole))

fourier_slanted = coeffs_to_fourier_slanted(coeffs, 0.25)
polynomial = coeffs_to_polynomial(coeffs)
poles = poles_from_polynomial(polynomial)

plot_poles(poles)
plot_fourier(fourier_slanted)

print("======= before slanted fourier clamp =======")
for pole in poles:
    print(np.absolute(pole))

fourier_slanted = clamp_fourier(fourier_slanted, 0.01)
coeffs = coeffs_from_fourier_slanted(fourier_slanted, 0.25)
polynomial = coeffs_to_polynomial(coeffs)
poles = poles_from_polynomial(polynomial)

plot_poles(poles)
plot_fourier(fourier_slanted)

print("======= after slanted fourier clamp =======")
for pole in poles:
    print(np.absolute(pole))
"""