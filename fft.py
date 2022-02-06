import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import csv
from matplotlib import mlab
from scipy import signal
from matplotlib import cm

input_name = "scope10"
audio_path = "./posnetki/" + input_name + ".wav"
osciloscop_path = "./meritve/" + input_name + ".csv"

cut_f_min = 40
cut_f_max = 60


def plot_magnitude_spectrum(signal, sr, title, min=0, max=120):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)

    plt.figure(figsize=(18, 5))

    f = np.linspace(0, sr, len(X_mag))

    plt.plot(f, X_mag)
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.xlim(min, max)

    return (f, X_mag)


# branje meritve iz CSV datoteke
file = open(osciloscop_path)
csvreader = csv.reader(file, delimiter=",")

rows = []
times = []
volts = []
count = 0
for row in csvreader:
    if count != 0:
        rows.append(row)
        times.append(float(row[0]))
        volts.append(float(row[1]))
    count += 1

SAMPLE_RATE = round(1 / abs(times[1] - times[0]))

audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
X = np.fft.fft(audio)

audio_f, audio_mag = plot_magnitude_spectrum(
    audio, sr, "FFT audio", cut_f_min, cut_f_max)
plt.show()


def specgram(x, fs, fmin = 0, fmax = 400):
  f, t, Sxx = signal.spectrogram(x, fs)
  freq_slice = np.where((f >= fmin) & (f <= fmax))
  f   = f[freq_slice]
  Sxx = Sxx[freq_slice,:][0]
  plt.pcolormesh(t, f, Sxx, shading='gouraud')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.show()


def specgram3d(x, fs=44100, ax=None, title=None, fmin = 0, fmax = 400):
    if not ax:
        ax = plt.axes(projection='3d')
        ax.set_title(title, loc='center', wrap=True)
        f, t, Sxx = signal.spectrogram(x, fs)
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f   = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0]
        
        X, Y, Z = t[None, :], f[:, None],  20.0 * np.log10(Sxx)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_ylabel('frequencies (Hz)')
        ax.set_ylabel('frequencies (Hz)')
        ax.set_zlabel('amplitude (dB)')
        # ax.set_zlim(-140, 0)
        return X, Y, Z

specgram(np.array(audio), SAMPLE_RATE, fmin = cut_f_min, fmax = cut_f_max)

fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
specgram3d(audio, srate=SAMPLE_RATE, fmin = cut_f_min, fmax = cut_f_max)
plt.show()