import numpy as np
import librosa

def __half_wave_rectify(signal):
    return np.maximum(signal, 0)

def __get_spectrum(signal, window_size=4096, hop_length=2048, gamma=1):
    """
    Compute the magnitude spectrum of an audio signal.
    Parameters
    ----------
        signal : np.ndarray [shape=(n,)]
            Audio signal.
        window_size : int > 0
            Size of the window.
        hop_length : int > 0
            Hop length.
    Returns
    -------
        spectrum : np.ndarray [shape=(n', m)]
            The magnitude spectrum.
    """
    stft = librosa.stft(signal, n_fft=window_size, hop_length=hop_length)
    power = np.abs(stft)**2
    # Apply Log compression
    power = 10 *np.log(1 + gamma * power)
    return power

def __get_local_avg(x, M=10):
    '''
    Compute the local average of the power spectrogram.
    Parameters
    ----------
        x : np.ndarray [shape=(n, m)]
            Input signal
        M : int > 0
            Size 2M+1 of the window to compute the local average.
    Returns
    -------
        local_avg : np.ndarray [shape=(n, m)]
            The local average of the input signal.
    '''
    # Compute the local average
    length = len(x)
    local_avg = np.zeros(length)
    for i in range(length):
        start = max(0, i - M)
        end = min(length, i + M + 1)
        local_avg[i] = np.mean(x[start:end])
    return local_avg

def getNovel(signal, fs,window_size=2048, hop_length=512, gamma=1, M=0, norm=False):
    """
    Compute the novelty function of an audio signal (Spectrum-based).
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
        novel : np.ndarray [shape=(n)]
            Novelty function.
    """
    fs_feature =  fs / hop_length
    # Compute Spectrum
    power = __get_spectrum(signal, window_size, hop_length, gamma)
    # Compute the difference between consecutive frames
    diff = power[:, 1:] - power[:, :-1]

    # Half-wave rectify the difference
    diff = __half_wave_rectify(diff)

    novel = np.sum(diff, axis=0)

    # Post Processing
    # Local average substraction
    if M > 0:
        local = __get_local_avg(novel)
        novel = np.maximum(novel - local, 0)
    # Max Normalization
    if norm:
        novel = novel / np.max(novel, 0)

    return novel, fs_feature