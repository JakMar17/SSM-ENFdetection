import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def fft(signal, sr):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    f = np.linspace(0, sr, len(X_mag))
    return (f, X_mag)


def plot_magnitude_spectrum(signal, sr, title, min=0, max=120):
    f, X_mag = fft(signal, sr)

    plt.plot(f, X_mag)
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.xlim(min, max)

    return (f, X_mag)


def specgram(x, fs, fmin=0, fmax=400, dB=False, title="Spektogram"):
    f, t, Sxx = signal.spectrogram(x, fs)
    freq_slice = np.where((f >= fmin) & (f <= fmax))
    f = f[freq_slice]
    Sxx = Sxx[freq_slice, :][0]

    if dB:
        Sxx = 20.0 * np.log10(Sxx)

    plt.title(title, loc='center', wrap=True)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()


def specgram3d(x, fs=44100, ax=None, title=None, fmin=0, fmax=400, dB=True):
    if not ax:
        ax = plt.axes(projection='3d')
        ax.set_title(title, loc='center', wrap=True)
        f, t, Sxx = signal.spectrogram(x, fs)
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice, :][0]
        if dB:
            Sxx = 20.0 * np.log10(Sxx)

        X, Y, Z = t[None, :], f[:, None],  Sxx
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_ylabel('frequencies (Hz)')
        ax.set_ylabel('frequencies (Hz)')
        if dB:
            ax.set_zlabel('amplitude (dB)')
        else:
            ax.set_zlabel('amplitude')
        # ax.set_zlim(-140, 0)
        return X, Y, Z


def enotski(v):
    return v / np.sqrt(np.sum(v**2))

def korelacija(data, sample):
  sample_l = len(sample)
  n = len(data) - len(sample)
#   print(n)

  corr = []

  for i in range(0, n):
    zac = i
    kon = zac + sample_l
    
    data_window = np.array(data[zac:kon])
    s = np.array(sample)

    c = np.correlate(enotski(data_window), enotski(s))
    corr.append(c)

  return corr


def time_duration(frequency, n):
    return n / frequency

def index_to_seconds(frequency, audio, index):
    audio_len = len(audio)
    audio_sec = time_duration(frequency, audio_len)
    return audio_sec / audio_len * index