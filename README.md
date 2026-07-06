🇧🇹 Deep learning models for Dzongkha language next-word prediction.

## Overview

This project implements three neural network architectures to predict the next word in Dzongkha text sequences:

- **LSTM**: 73.79% accuracy
- **Bi-LSTM**: 71.14% accuracy  
- **GRU**: 74.03% accuracy ⭐ (Best)

## Key Statistics

| Metric                | Value                      |
|-----------------------|----------------------------|
| **Best Model**        | GRU with 74.03% accuracy   |
| **Dataset**           | 100,000 Dzongkha sentences |
| **Total Words**       | 1,331,282                  |
| **Unique Vocabulary** | 28,344 words               |
| **Inference Speed**   | 65ms (CPU) / 10ms (GPU)    |

## Installation

```bash
# Clone the repository
git clone https://github.com/PrernaChhetri/Dzongkha-Next-Word-Prediction.git
cd Dzongkha-Next-Word-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Models

### LSTM Model
- Accuracy: 73.79%
- Training time: 1 hour
- File: `LSTM_for_nextword_Version6.py`

### Bi-LSTM Model
- Accuracy: 71.14%
- Training time: 30 minutes
- File: `BiLSTM_PHUNTSHO.py`

### GRU Model (BEST)
- Accuracy: 74.03%
- Training time: 17 minutes
- File: `GRU_version13.py`

## Usage

See Jupyter notebooks in the `notebooks/` folder for training examples.

## 📜 License

MIT License

## 🚀 Status

✅ Production Ready | 📊 Peer-Reviewed | 🎓 Academic Project
