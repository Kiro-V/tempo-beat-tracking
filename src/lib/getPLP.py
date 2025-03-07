import numpy as np
import librosa

def getPLP(X, Fs, L, win_length = 2048, hop_size = 512, Theta = np.arange(50, 201, 1)):
    '''
    Compute the PLP of an audio signal.
    Parameters
    ----------
        X : np.ndarray [shape=(n, m)]
            Complex Fourier Tempogram
        Fs : int > 0
            Sampling rate of the signal.
        L : int > 0
            Frame parameter of signal.
        win_length : int > 0
            Size of the window.
        hop_size : int > 0
            Hop length.
        Theta : np.ndarray [shape=(n',)]
            Tempo Axis (BPM).
    Returns
    -------
        nov_PLP : np.ndarray [shape=(n', m)]
            Resulting PLP.
    '''
    win = np.hanning(win_length)
    N_left = win_length // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    nov_PLP = np.zeros(L_pad)
    M = X.shape[1]
    tempogram = np.abs(X)

    for m in range(M):
        k = np.argmax(tempogram[:,m])
        tempo = Theta[k]
        omega = (tempo / 60) / Fs
        c = X[k, m]
        phase = -np.angle(c) / (2 * np.pi)
        start = m * hop_size
        end = start + win_length
        t_kernel = np.arange(start, end)
        kernel = win * np.cos(2 * np.pi * (t_kernel * omega - phase))
        nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel
    nov_PLP = nov_PLP[L_left:-L_right]
    nov_PLP = np.maximum(nov_PLP, 0)
    return nov_PLP