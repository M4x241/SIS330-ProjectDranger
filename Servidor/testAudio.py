import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class AudioCRNN(nn.Module):
    def __init__(self):
        super(AudioCRNN, self).__init__()
        # Extractor Espacial/Frecuencial: Red Convolucional (CNN)
        # Espera entrada con shape: (Batch, Channels=1, Mels=64, Time)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Después de 4 iteraciones de MaxPool 2x2, la dimensión Frecuencial de 64 Mels bajó a:
        # 64 / (2^4 = 16) = 4 Mels residuales por frame.
        # Las características (Features) totales por bloque de tiempo serán: 64 Canales * 4 = 256.
        
        # Extractor Temporal: Red Neuronal Recurrente (RNN - LSTM)
        self.rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Clasificador Multicapa (Fully Connected)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Salida logit (sin Sigmoid aquí ya que usamos BCEWithLogitsLoss luego)
        )

    def forward(self, x):
        # Paso por la CNN
        x = self.cnn(x)
        
        # Preparación del tensor para la RNN
        # x actualmente tiene formato: (Batch, Canales, Frecuencias, Tiempo)
        # Lo modificamos a: (Batch, Tiempo, Canales, Frecuencias)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, mels_features = x.shape
        # Achatamos las Frecuencias y Canales
        x = x.reshape(batch_size, time_steps, channels * mels_features)
        
        # Paso temporal a través de la RNN
        rnn_out, (hidden_state, cell_state) = self.rnn(x)
        
        # Nos interesa solo el último hidden state de la última capa como representación de todo el audio
        last_hidden = hidden_state[-1] 
        
        # Clasificación binaria (es motosierra o no)
        out = self.fc(last_hidden)
        return out
import torch
import torchaudio
import torch.nn as nn
import numpy as np

# Copia tu clase AudioCRNN aquí pero con esta pequeña corrección en forward:
# out = self.fc(rnn_out[:, -1, :])  <-- Esto es más seguro que usar hidden_state

def test_audio_corregido(file_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar modelo
    model = AudioCRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Cargar audio con Torchaudio (Misma lógica que el entrenamiento)
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3. Transformaciones EXACTAS del entrenamiento
    # En tu entrenamiento usaste MelSpectrogram(16000, 64, 1024, 512)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
    ).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

    with torch.no_grad():
        spec = mel_transform(waveform.to(device))
        spec_db = db_transform(spec) # (1, 64, tiempo)
        
        # 4. Inferencia
        # Tu CNN espera (Batch, 1, Mels, Time)
        input_tensor = spec_db.unsqueeze(0) 
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    print(f"\nAnalizando: {file_path}")
    print(f"Probabilidad detectada: {probability:.2%}")
    if probability > 0.5:
        print("🚨 RESULTADO: ¡MOTOSIERRA!")
    else:
        print("🌲 RESULTADO: BOSQUE SEGURO")

if __name__ == "__main__":
    MODEL = "/home/ghost/Documents/Proyectos/python/Proyecto Python/detector_motosierra_crnn.pth"
    AUDIO = "/home/ghost/Documents/Proyectos/python/Proyecto Python/Servidor/audiopapkin-chainsaw-297887.mp3"
    test_audio_corregido(AUDIO, MODEL)