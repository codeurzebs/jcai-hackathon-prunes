# ===============================================================
# JCIA HACKATHON 2025 - TRI INTELLIGENT DES PRUNES AFRICAINES 🍑
# Auteurs : ZEBS HAUPUR & TIOJIO ROMAIN
# Description : Projet d’IA pour la classification automatique
#               des prunes selon leur qualité visuelle.
# Technologies : PyTorch | Streamlit | Azure ML | Python
# GitHub : https://github.com/NGcodeX/jcai-hackathon-prunes
# Tel: +237692077005
# Communauté: NGcodeX
# ===============================================================

import torch.nn as nn

class PlumClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlumClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ===============================================================
# JCIA HACKATHON 2025 - TRI INTELLIGENT DES PRUNES AFRICAINES 🍑
# Auteurs : ZEBS HAUPUR & TIOJIO ROMAIN
# Description : Projet d’IA pour la classification automatique
#               des prunes selon leur qualité visuelle.
# Technologies : PyTorch | Streamlit | Azure ML | Python
# GitHub : https://github.com/NGcodeX/jcai-hackathon-prunes
# Tel: +237692077005
# Communauté: NGcodeX
# ===============================================================
