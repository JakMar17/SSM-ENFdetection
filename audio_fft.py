import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np

audio_path = "./posnetki/scope10.wav"
print(librosa.get_samplerate(audio_path))

audio, sr = librosa.load(audio_path)

X = np.fft.fft(audio)
len(X)

def plot_magnitude_spectrum(signal, sr, title, min = 0, max = 120):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    
    plt.figure(figsize=(18, 5))
    
    f = np.linspace(0, sr, len(X_mag))
    
    plt.plot(f, X_mag)
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.xlim(min, max)


audio_avg = np.average(audio)
audio_x = []
for a in audio:
  audio_x.append(a - audio_avg)

# plot_magnitude_spectrum(audio_x, sr, "violin", 45,55)
# plt.show()


D = librosa.stft(audio)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
ax.set(title='Using a logarithmic frequency axis')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()

mes_f, mes_mag = plot_magnitude_spectrum(volts, SAMPLE_RATE, "violin", cut_f_min, cut_f_max)