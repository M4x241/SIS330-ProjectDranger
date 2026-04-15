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
    
    
import socket
import numpy as np
import torch
import torchaudio
import torch.nn as nn

# --- CONFIGURACIÓN ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MODEL_PATH = "/home/ghost/Documents/Proyectos/python/Proyecto Python/ModelSounds/Results/detector_motosierra_crnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Cargar modelo una sola vez
model = AudioCRNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 2. Definir transformaciones (Idénticas al entrenamiento)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
).to(device)
db_transform = torchaudio.transforms.AmplitudeToDB().to(device)
# Resampler: Unity(44.1k/48k) -> Modelo(16k). Ajusta 44100 si tu Unity usa otra.
resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000).to(device)

# --- SOCKET Y BUFFER ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

audio_buffer = []
# Necesitamos aprox 1 segundo de audio (44100 muestras) para una buena detección
UMBRAL_MUESTRAS = 44100 

print(f"Servidor listo en GPU: {torch.cuda.is_available()}")

while True:
    data, addr = sock.recvfrom(65535) # Buffer de red grande
    chunk = np.frombuffer(data, dtype=np.float32)
    audio_buffer.extend(chunk)

    if len(audio_buffer) >= UMBRAL_MUESTRAS:
        # Convertir buffer a tensor
        waveform = torch.tensor(audio_buffer, dtype=torch.float32).unsqueeze(0).to(device)
        audio_buffer = [] # Limpiar buffer para el siguiente segundo

        with torch.no_grad():
            # 1. Resamplear a 16kHz
            waveform_16k = resampler(waveform)
            
            # 2. Espectrograma de Mel en DB
            spec = mel_transform(waveform_16k)
            spec_db = db_transform(spec)
            
            # 3. Inferencia (Batch, Channel, Mels, Time)
            input_tensor = spec_db.unsqueeze(0)
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

        if probability > 0.5:
            print(f"🚨 ¡ALERTA! Motosierra detectada: {probability:.2%}")
        else:
            print(f"🌲 Bosque en paz: {probability:.2%}")