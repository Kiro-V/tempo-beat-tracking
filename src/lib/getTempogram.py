import numpy as np
from scipy.interpolate import interp1d

def getTempogram(x, Fs,  win_length = 2048, hop_size = 512, Theta = np.arange(50, 201, 1), mode = 'autocorr', norm = False):
    '''
    Compute the tempogram of an audio signal.
    Parameters
    ----------
        signal : np.ndarray [shape=(n,)]
            Audio signal.
        Fs : int > 0
            Sampling rate of the audio signal.
        window_size : int > 0
            Size of the window
        hop_length : int > 0
            Hop length
        Theta: np.ndarray [shape=(n',)]
            Tempo Range (BPM).
        mode : str in ['autocorr', 'fourier']
            Mode of computation.
        norm : bool
            Normalize the tempogram (only for autocorr mode).
    Returns
    -------
        tempogram : np.ndarray [shape=(n', m)]
            Tempogram.
        T_coef : np.ndarray [shape=(m,)]
            Time axis (seconds).
        F_coef_BPM : np.ndarray [shape=(n',)]
            Tempo Axis (BPM).
    '''
    assert mode in ['autocorr', 'fourier'], 'Mode must be either autocorr or fourier'

    N_left = win_length // 2 # Left side of the window
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(0, L_pad)
    M = int(np.floor((L_pad - win_length) / hop_size)) + 1
    if mode == 'fourier':
        K = len(Theta)
        X = np.zeros((K, M), dtype=np.complex_)
        win = np.hanning(win_length) # Hanning Window
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
    elif mode == 'autocorr':
        tempo_min = Theta[0]
        tempo_max = Theta[-1]
        lag_min = int(np.ceil(Fs * 60 / tempo_max))
        lag_max = int(np.ceil(Fs * 60 / tempo_min))
        # Calculate the tempogram
        X = np.zeros((win_length, M))
        win = np.ones(win_length) # Rectangular Window
        if norm:
            lag_sum = np.arange(win_length, 0, -1)
        for m in range(M):
            start = m * hop_size
            end = start + win_length
            x_local = win * x_pad[start:end]
            r_xx = np.correlate(x_local, x_local, mode='full')
            r_xx = r_xx[win_length - 1:]
            if norm:
                r_xx = r_xx / lag_sum
            X[:, m] = r_xx
        T_coef = np.arange(0, M) * hop_size / Fs
        F_coef_lag = np.arange(0, win_length) / Fs
        # Extract the part within the tempo range
        X_cut = X[lag_min:lag_max+1, :]
        F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
        F_coef_BPM_cut = 60 / F_coef_lag_cut
        tempogram = interp1d(F_coef_BPM_cut, X_cut, kind='linear', axis=0, fill_value='extrapolate')(Theta)

        return tempogram, T_coef, Theta, X_cut, F_coef_lag_cut