import numpy as np

def getTempogram(x, Fs,  win_length = 2048, hop_size = 512, Theta = np.arange(50, 201, 1)):
    '''
    Compute the tempogram of an audio signal.
    Parameters
    ----------
        signal : np.ndarray [shape=(n,)]
            Audio signal.
        window_size : int > 0
            Size of the window
        hop_length : int > 0
            Hop length
        gamma: float > 0
            Log compression factor.
    Returns
    -------
        tempogram : np.ndarray [shape=(n', m)]
            Tempogram.
        T_coef : np.ndarray [shape=(m,)]
            Time axis (seconds).
        F_coef_BPM : np.ndarray [shape=(n',)]
            Tempo Axis (BPM).
    '''
    win = np.hanning(win_length) # Hanning Window
    N_left = win_length // 2 # Left side of the window
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(0, L_pad)
    M = int(np.floor((L_pad - win_length) / hop_size)) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] /60 ) / Fs
        exponential = np.exp(-2j * np.pi * omega * t_pad)
        x_exp = x_pad * exponential
        for m in range(M):
            start = m * hop_size
            end = start + win_length
            X[k, m] = np.sum(x_exp[start:end] * win)
        T_coef = np.arange(0, M) * hop_size / Fs
        F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM