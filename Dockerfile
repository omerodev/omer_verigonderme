# 1) Küçük bir Python imajı
FROM python:3.11-slim

# 2) Sistemi güncelle, OpenCV vb. kütüphaneleri kur
RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

# 3) Çalışma dizinini ayarla
WORKDIR /app

# 4) Kodunu ve indirme script’ini kopyala
COPY . /app

# 5) ML modellerini indir
RUN bash download_models.sh

# 6) Python bağımlılıklarını kur
RUN pip install --no-cache-dir -r requirements.txt

# 7) Django migrate & collectstatic
RUN python3 manage.py migrate --noinput
RUN python3 manage.py collectstatic --noinput

# 8) Environment değişkenleri
ENV DJANGO_SETTINGS_MODULE=kontrolsitesi.settings \
    PYTHONUNBUFFERED=1

# 9) Port ayarı (Spaces 7860’i ya da 80’i de kullanır)
EXPOSE 8000

# 10) Uygulamayı Gunicorn ile çalıştır
CMD ["gunicorn", "kontrolsitesi.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "1", "--timeout", "300"]
