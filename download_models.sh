#!/usr/bin/env bash
set -e

# İndirmeleri koyacağımız klasörü oluştur
mkdir -p wifimodule/ml_models

# 1) classes.json
curl -L -o wifimodule/ml_models/classes.json \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/classes.json?download=true"

# 2) classes50.json
curl -L -o wifimodule/ml_models/classes50.json \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/classes50.json?download=true"

# 3) hayvanbeit1.pth
curl -L -o wifimodule/ml_models/hayvanbeit1.pth \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/hayvanbeit1.pth?download=true"

# 4) my_face_ref.npy
curl -L -o wifimodule/ml_models/my_face_ref.npy \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/my_face_ref.npy?download=true"

# 5) quickdraw502.pth
curl -L -o wifimodule/ml_models/quickdraw502.pth \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/quickdraw502.pth?download=true"

# 6) resim_cizen.pth
curl -L -o wifimodule/ml_models/resim_cizen.pth \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/resim_cizen.pth?download=true"

# 7) yolo_face_detection.pt
curl -L -o wifimodule/ml_models/yolo_face_detection.pt \
     "https://huggingface.co/omeromeromer/omer-verigonderme-models/resolve/main/yolo_face_detection.pt?download=true"

echo "Tüm modeller indirildi → wifimodule/ml_models/"
