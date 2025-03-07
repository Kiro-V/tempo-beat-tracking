import numpy as np
import matplotlib.pyplot as plt
import scipy

def __get_extent(T_coef, F_coef):
    x_ext1 = (T_coef[1] - T_coef[0]) / 2
    x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
    y_ext1 = (F_coef[1] - F_coef[0]) / 2
    y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
    extent = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    return extent

def plot_tempogram(X, T_coef, F_coef, title='Tempogram', cmap='inferno', fit_pulse=False):
    plt.figure(figsize=(10, 4))
    extent = __get_extent(T_coef, F_coef)
    plt.imshow(np.abs(X), aspect='auto', origin='lower', extent=extent, cmap=cmap)
    if fit_pulse:
        for m in range(X.shape[1]):
            k = np.argmax(X[:,m])
            plt.plot(T_coef[m], F_coef[k], 'bo')
    plt.xlabel('Time (s)')
    plt.ylabel('Tempo (BPM)')
    plt.title(title)
    plt.colorbar()

def plot_curve(novel, Fs, title='Novelty Function', get_peaks=False):
    plt.figure(figsize=(10, 4))
    time_pos = np.arange(0, len(novel)) / Fs
    plt.plot(time_pos, novel)
    if get_peaks:
        peaks = scipy.signal.find_peaks(novel, prominence=0.02)[0]
        plt.plot(peaks / Fs, novel[peaks], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Novelty')
    plt.title(title)