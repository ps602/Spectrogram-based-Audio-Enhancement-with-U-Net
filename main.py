import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import argparse
from utils import *
from models import *

parser = argparse.ArgumentParser(description="something")
parser.add_argument('--model_name', type=str, default='unet')
parser.add_argument('--method', type=str, default='eval')
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--n_mels', type=int, default=128)
args = parser.parse_args()


# Read Audio Data
audio_path = "/home/priya/ADI/data/eval/audio.wav"
y, sr = librosa.load(audio_path) 

# Create Spectrogram
n_mels = args.n_mels  # Number of Mel bands to generate
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
S_dB = librosa.power_to_db(S, ref=np.max)
spectrogram_tensor = torch.from_numpy(S_dB).unsqueeze(0).unsqueeze(0).float()

# Choosing the Model
if args.model_name == "unet":
    model = Unet(n_channels=1,n_classes=1)
    if args.method == "eval":
        model.eval()
        with torch.no_grad():
            output = model(spectrogram_tensor)
    else:
        model.train_model()
else:
    model = UNetWithTimeShift()
    if args.method == "eval":
        model.eval()
        with torch.no_grad():
            output = model(spectrogram_tensor)
    else:
        model.train_model()

# Visualizing Inputs and Outputs
if args.method == 'eval':
    
    plt.title('Audio x Time Plot')
    librosa.display.waveshow(y, sr=sr)
    plt.savefig('results/input_wave.png')
    
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('results/input_spec.png')

    mel_spectrogram = output.detach().cpu().numpy()
    mel_spectrogram = mel_spectrogram.squeeze()
    ## Plotting the Output tensor as a Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed Mel Spectrogram')
    plt.tight_layout()
    plt.figsave(f'results/output_{args.model_name}.png')
    plt.show()
    
    ##Plotting the input Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input Mel Spectrogram')
    plt.tight_layout()
    plt.figsave(f'results/output_{args.model_name}.png')
    plt.show()
else:
    pass
