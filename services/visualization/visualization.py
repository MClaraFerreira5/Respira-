import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from core.config import SAMPLE_RATE, CLASS_INDICES

matplotlib.use('TkAgg')


def generate_spectrogram(audio, prediction=None, save_path=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.specgram(audio, Fs=SAMPLE_RATE, cmap=cm.viridis, NFFT=512, noverlap=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma do Áudio')

    plt.subplot(3, 1, 2)
    time_axis = np.linspace(0, len(audio) / SAMPLE_RATE, num=len(audio))
    plt.plot(time_axis, audio)
    plt.title('Forma de Onda')

    if prediction is not None:
        plt.subplot(3, 1, 3)
        classes = list(CLASS_INDICES.keys())
        values = [prediction[idx] for idx in CLASS_INDICES.values()]
        plt.bar(classes, values)
        plt.title('Probabilidades das Classes')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
