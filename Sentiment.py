#!pip install torch torchtext nltk matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

nltk.download('stopwords')

# Konfiguracja
SEED = 42
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
torch.manual_seed(SEED)

# Przetwarzanie tekstu
TEXT = data.Field(
    tokenize='spacy',
    lower=True,
    stop_words=set(stopwords.words('english')),
    preprocessing=lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
)
LABEL = data.LabelField(dtype=torch.float)

# Wczytanie danych IMDb
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Budowa słownika
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# Podział na batche
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Model LSTM
class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(self.dropout(hidden.squeeze(0)))

model = SentimentLSTM().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Trening
losses, accuracies = [], []
for epoch in range(5):
    total_loss = 0
    correct = 0
    
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += ((torch.sigmoid(predictions) > 0.5).float().eq(batch.label).sum().item()
    
    accuracy = correct / len(train_data)
    losses.append(total_loss / len(train_iterator))
    accuracies.append(accuracy)
    print(f'Epoch {epoch+1}: Loss {losses[-1]:.2f}, Accuracy {accuracy:.2%}')

# Wizualizacja
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(losses, label='Loss')
plt.subplot(1,2,2)
plt.plot(accuracies, label='Accuracy')
plt.show()

# Macierz pomyłek
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_iterator:
        predictions = torch.sigmoid(model(batch.text).squeeze(1))
        y_true.extend(batch.label.cpu().numpy())
        y_pred.extend((predictions > 0.5).cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.show()