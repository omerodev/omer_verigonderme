from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
'''
def home(request):
    received_text = None

    if request.method == "POST":
        received_text = request.POST.get('text_input')  # Formdan gelen veriyi al

    return render(request, 'home.html', {'received_text': received_text})
'''
from django.shortcuts import render
from django.http import JsonResponse
'''
@csrf_exempt
def home(request):
    received_text = None
    if request.method == "POST":
        received_text = request.POST.get('text_input')  # Formdan gelen veriyi al

    return render(request, 'home.html', {'received_text': received_text})
@csrf_exempt
def get_text(request):
    # ESP8266 bu yola GET isteği yaparak veriyi alacak
    received_text = request.GET.get('text', '')
    return JsonResponse({'text': received_text})
'''

# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse

from django.shortcuts import render
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt

from .models import TextData
from django.shortcuts import render
from django.http import JsonResponse
from .models import TextData
from django.views.decorators.csrf import csrf_exempt
from .models import DistanceData
from .models import TemperatureData, HumidityData, DistanceData
# Verileri kaydetme ve gösterme fonksiyonu
@csrf_exempt
def home(request):
    received_text = None
    if request.method == "POST":
        # Formdan gelen verileri al
        text_input = request.POST.get('text_input')
        sayi_input = request.POST.get('sayi_input')
        selected_button = request.POST.get('button')

        # Veritabanına metin ve sayı değerini kaydet
        if text_input:
            TextData.objects.create(text_input=text_input)
            received_text = text_input
        elif sayi_input:
            TextData.objects.create(text_input=f"asayi:{sayi_input}")
            received_text = f"asayi:{sayi_input}"
        elif selected_button:
            TextData.objects.create(text_input=selected_button)
            received_text = selected_button

    return render(request, 'home.html', {'received_text': received_text})

# ESP8266'nın veriyi çekeceği fonksiyon
@csrf_exempt
def get_text(request):
    # En son kaydedilen text verisini al
    latest_text = TextData.objects.last()
    if latest_text:
        return JsonResponse({'text': latest_text.text_input})
    else:
        return JsonResponse({'text': 'Veri yok'})




import datetime
last_data_time = None

@csrf_exempt
def uzaklik_olcumu(request):
    global last_data_time

    # Eğer istek AJAX isteği ise sadece yeni veriyi kontrol edip JSON yanıt dön
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        latest_data = DistanceData.objects.last()
        if latest_data and latest_data.timestamp != last_data_time:
            last_data_time = latest_data.timestamp
            return JsonResponse({'new_data': True})
        return JsonResponse({'new_data': False})

    # Normal GET isteğiyle sayfayı yüklemek için
    all_data = DistanceData.objects.all().order_by('timestamp')
    latest_data = all_data.last()
    if latest_data:
        last_data_time = latest_data.timestamp

    return render(request, 'uzaklik_olcumu.html', {
        'all_data': all_data,
        'latest_data': latest_data.distance if latest_data else "Veri yok",
    })

@csrf_exempt
def add_distance_data(request):
    if request.method == "POST":
        distance = request.POST.get('distance')
        if distance:
            DistanceData.objects.create(distance=float(distance), timestamp=datetime.datetime.now())
            return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})



last_temp_time = None
last_humidity_time = None

@csrf_exempt
def sicaklik_olcumu(request):
    global last_temp_time

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        latest_data = TemperatureData.objects.last()
        if latest_data and latest_data.timestamp != last_temp_time:
            last_temp_time = latest_data.timestamp
            return JsonResponse({'new_data': True})
        return JsonResponse({'new_data': False})

    all_data = TemperatureData.objects.all().order_by('timestamp')
    latest_data = all_data.last()
    if latest_data:
        last_temp_time = latest_data.timestamp

    return render(request, 'sicaklik_olcumu.html', {
        'all_data': all_data,
        'latest_data': latest_data.temperature if latest_data else "Veri yok",
    })

@csrf_exempt
def nem_olcumu(request):
    global last_humidity_time

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        latest_data = HumidityData.objects.last()
        if latest_data and latest_data.timestamp != last_humidity_time:
            last_humidity_time = latest_data.timestamp
            return JsonResponse({'new_data': True})
        return JsonResponse({'new_data': False})

    all_data = HumidityData.objects.all().order_by('timestamp')
    latest_data = all_data.last()
    if latest_data:
        last_humidity_time = latest_data.timestamp

    return render(request, 'nem_olcumu.html', {
        'all_data': all_data,
        'latest_data': latest_data.humidity if latest_data else "Veri yok",
    })

@csrf_exempt
def add_temperature_data(request):
    if request.method == "POST":
        temperature = request.POST.get('temperature')
        if temperature:
            TemperatureData.objects.create(temperature=float(temperature), timestamp=datetime.datetime.now())
            return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

@csrf_exempt
def add_humidity_data(request):
    if request.method == "POST":
        humidity = request.POST.get('humidity')
        if humidity:
            HumidityData.objects.create(humidity=float(humidity), timestamp=datetime.datetime.now())
            return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})



# views.py
from .models import BuzzerData

@csrf_exempt
def toggle_buzzer(request):
    if request.method == "POST":
        status = request.POST.get('status')
        if status:
            BuzzerData.objects.create(status=status)
            return JsonResponse({'status': 'success', 'current_status': status})
    else:
        # En son buzzer durumunu döndür
        latest_status = BuzzerData.objects.last()
        if latest_status:
            return JsonResponse({'status': 'success', 'current_status': latest_status.status})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})



# views.py
from django.shortcuts import render
from .models import Location

def show_map(request):
    # Veritabanındaki en son eklenmiş konumu al
    latest_location = Location.objects.last()
    context = {
        "latitude": latest_location.latitude if latest_location else 0,
        "longitude": latest_location.longitude if latest_location else 0,
    }
    return render(request, 'show_map.html', context)




# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Location

@csrf_exempt
def save_location(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        # Veritabanına konum kaydet
        Location.objects.create(latitude=latitude, longitude=longitude)
        return JsonResponse({"status": "success", "message": "Konum kaydedildi."})

    return JsonResponse({"status": "fail", "message": "Geçersiz istek."})




import requests
from django.shortcuts import render
from .models import Location

def get_weather_icon(weathercode):
    """Hava durumu koduna göre uygun Font Awesome ikon sınıfını döndür."""
    if weathercode == 0:
        return "fas fa-sun"
    elif weathercode == 1:
        return "fas fa-cloud"
    elif weathercode == 2:
        return "fas fa-cloud-sun"
    elif weathercode == 3:
        return "fas fa-cloud-showers-heavy"
    elif weathercode in [45, 48]:
        return "fas fa-smog"
    elif weathercode in [51, 53, 55]:
        return "fas fa-cloud-rain"
    elif weathercode in [61, 63, 65]:
        return "fas fa-cloud-showers-heavy"
    elif weathercode in [71, 73, 75]:
        return "fas fa-snowflake"
    elif weathercode == 95:
        return "fas fa-bolt"
    else:
        return "fas fa-cloud"

def hava_durumu(request):
    # Veritabanındaki son konumu al
    latest_location = Location.objects.last()

    # Eğer konum yoksa varsayılan enlem ve boylam
    latitude = latest_location.latitude if latest_location else 0
    longitude = latest_location.longitude if latest_location else 0

    # Open-Meteo API isteği
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,windspeed_10m_max,weathercode&timezone=auto"
    response = requests.get(api_url)
    weather_data = response.json()

    # API'den gelen hava durumu bilgisi
    current_weather = weather_data.get("current_weather", {})
    daily_forecast = weather_data.get("daily", {})

    # 5 günlük tahmin için veri hazırlama
    forecast = []
    for i in range(len(daily_forecast.get("time", []))):
        forecast.append({
            "date": daily_forecast["time"][i],
            "temperature": daily_forecast["temperature_2m_max"][i],
            "windspeed": daily_forecast["windspeed_10m_max"][i],
            "weather_icon": get_weather_icon(daily_forecast["weathercode"][i])
        })

    context = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": current_weather.get("temperature", "Bilinmiyor"),
        "windspeed": current_weather.get("windspeed", "Bilinmiyor"),
        "weather_icon": get_weather_icon(current_weather.get("weathercode", 0)),
        "forecast": forecast,
    }

    return render(request, 'havadurumu.html', context)










from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import base64
import datetime
from .models import CameraImage

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        filename = f"esp32_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image = CameraImage(image=image_file, name=filename)
        image.save()
        return JsonResponse({"status": "success", "message": "Görüntü kaydedildi."})
    return JsonResponse({"status": "error", "message": "Geçersiz istek."})



import os
import cv2
import numpy as np
import torch

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

from .models import CameraImage, TextData, DistanceData, TemperatureData, HumidityData, BuzzerData, Location

# ————————————
# 1) Face-recognition modellerini bir kez yükleyelim
# ————————————
MODEL_DIR       = os.path.join(settings.BASE_DIR, "wifimodule", "ml_models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo_face_detection.pt")
EMBED_PATH      = os.path.join(MODEL_DIR, "my_face_ref.npy")

yolo    = YOLO(YOLO_MODEL_PATH)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
ref_emb = np.load(EMBED_PATH)
THRESHOLD = 0.8

# views.py en üstünde (yolo/facenet tanımlamalarının hemen altına)

import json
import timm
from torchvision import transforms
from PIL import Image

# ——— BeiT hayvan sınıflandırma modeli ———
MODEL_DIR        = os.path.join(settings.BASE_DIR, "wifimodule", "ml_models")
BEIT_MODEL_PATH  = os.path.join(MODEL_DIR, "hayvanbeit1.pth")
CLASSES_PATH     = os.path.join(MODEL_DIR, "classes.json")

# Sınıf isimleri
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    animal_classes = json.load(f)
device = torch.device("cpu")
# Modeli yükle
beit_model = timm.create_model(
    'beit_base_patch16_224',
    pretrained=False,
    num_classes=len(animal_classes)
)
beit_model.load_state_dict(torch.load(BEIT_MODEL_PATH, map_location=device))
beit_model.to(device).eval()

# Ön işleme dönüşümleri (Tkinter kodunuzdakilerle aynı)
beit_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ————————————
# 2) Kamera akışı (video_feed)
# ————————————
def video_feed(request):
    """
    Son kaydedilmiş CameraImage’ı döner.
    Eğer en son TextData.text_input == "camyuz" ise,
    önce yüz tespiti+tanıma yapıp kutulu, etiketli JPEG olarak gönderir.
    """
    latest_img = CameraImage.objects.last()
    if not latest_img:
        return HttpResponse("Görüntü Yok", content_type="text/plain")

    img_path = latest_img.image.path
    last_text_obj = TextData.objects.last()
    last_text = last_text_obj.text_input if last_text_obj else ""

    if last_text.lower() == "camyuz":
        img = cv2.imread(img_path)
        if img is not None:
            # Parametreler
            box_thickness   = 3
            font            = cv2.FONT_HERSHEY_SIMPLEX
            font_scale      = 0.5
            font_thickness  = 1
            padding         = 5
            text_color      = (255,255,255)

            results = yolo.predict(source=img, conf=0.5)
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())

                    # Embedding + mesafe hesaplama
                    crop = img[y1:y2, x1:x2]
                    face = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (160,160))
                    t = torch.tensor(face).permute(2,0,1).unsqueeze(0).float()/255.0
                    with torch.no_grad():
                        emb = facenet(t)[0].cpu().numpy()
                    dist  = np.linalg.norm(emb - ref_emb)
                    label = "OMER" if dist < THRESHOLD else "Bilinmeyen"
                    color = (0,255,0) if label=="OMER" else (0,0,255)

                    # 1) Kalın çerçeve
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

                    # 2) Metin boyutu ve arka plan
                    text = f"{label}:{dist:.2f}"
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    bg_tl = (x1, y1 - th - baseline - padding)
                    bg_br = (x1 + tw + padding, y1)
                    cv2.rectangle(img, bg_tl, bg_br, color, cv2.FILLED)

                    # 3) Etiketi çiz
                    text_org = (x1 + padding//2, y1 - baseline - padding//2)
                    cv2.putText(img, text, text_org, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            ret, buf = cv2.imencode('.jpg', img)
            return HttpResponse(buf.tobytes(), content_type="image/jpeg")

    elif last_text == "camhayvan":
        # OpenCV ile oku
        img = cv2.imread(img_path)
        if img is not None:
            # PIL formatına çevirip inference
            pil = Image.open(img_path).convert("RGB")
            inp = beit_transforms(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = beit_model(inp)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            conf = float(probs[idx])

            # Eşik altındaysa “Hayvan yok”
            if conf < 0.5:
                text = "Hayvan yok"
                box_color = (0, 0, 255)
            else:
                text = f"{animal_classes[idx]} {conf * 100:.1f}%"
                box_color = (0, 255, 0)

            # Resim boyutları
            h, w, _ = img.shape

            # 1) Üstte dolgu kutusu
            cv2.rectangle(img, (0, 0), (w, 40), box_color, cv2.FILLED)
            # 2) Etiketi yaz
            cv2.putText(
                img, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,  # font_scale
                (255, 255, 255),  # beyaz yazı
                2,  # kalınlık
                cv2.LINE_AA
            )

            # JPEG kodlayıp dön
            ret, buf = cv2.imencode('.jpg', img)
            return HttpResponse(buf.tobytes(), content_type="image/jpeg")


    # Komut farklıysa ham resmi dön
    with open(img_path, "rb") as f:
        return HttpResponse(f.read(), content_type="image/jpeg")


from django.shortcuts import render

def canli_goruntu(request):
    return render(request, 'video_feed.html')







import os
import io
import uuid
import base64
import torch

from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import render
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageOps

from .models import Drawing

# --- Classification model imports ---

# --- 50-class model imports ---
from .ml50 import predict_file50, classes50
# --- Generation model imports ---
from .mlgenerate import SketchGenCond, sample_sequence, CLASSES as GEN_CLASSES

# Device
DEVICE = torch.device("cpu")

# Load generation model
APP_DIR = os.path.dirname(__file__)
GEN_MODEL_PATH = os.path.join(APP_DIR, "ml_models", "resim_cizen.pth")

gen_model = SketchGenCond().to(DEVICE)
if not os.path.exists(GEN_MODEL_PATH):
    raise FileNotFoundError(f"Generation model not found: {GEN_MODEL_PATH!r}")
gen_model.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE))
gen_model.eval()

def home50(request):
    saved_image_url = None
    prediction50 = None
    processed64 = None
    probabilities50 = None

    if request.method == "POST":
        fmt, imgstr = request.POST["image"].split(";base64,")
        img_bytes = base64.b64decode(imgstr)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        buf1 = io.BytesIO()
        img.resize((280, 280), Image.Resampling.LANCZOS).save(buf1, format="PNG")
        buf1.seek(0)
        name = f"drawing_{uuid.uuid4()}.png"
        drawing = Drawing.objects.create(
            image=ContentFile(buf1.read(), name=name)
        )
        saved_image_url = drawing.image.url

        pil64 = Image.open(drawing.image.path).convert("L").resize((64, 64), Image.Resampling.NEAREST)
        pil64 = ImageOps.invert(pil64)

        buf2 = io.BytesIO()
        pil64.save(buf2, format="PNG")
        processed64 = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode()

        prediction50, probs50 = predict_file50(drawing.image.path)
        probabilities50 = sorted(zip(classes50, probs50.tolist()), key=lambda x: x[1], reverse=True)

    return render(request, "home50.html", {
        "saved_image_url": saved_image_url,
        "processed64": processed64,
        "prediction50": prediction50,
        "probabilities50": probabilities50,
    })


def generate(request):
    generated_image = None

    if request.method == "POST":
        cls_id = int(request.POST.get("class_id", 0))
        seq = sample_sequence(gen_model, cls_id)

        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        x = y = 0
        for dx, dy, p1, p2, p3 in seq:
            nx, ny = x + dx, y + dy
            if p1:
                ax.plot([x, nx], [y, ny], linewidth=2)
            x, y = nx, ny

        ax.invert_yaxis()
        ax.axis("off")

        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        generated_image = f"data:image/png;base64,{b64}"

    return render(request, "generate.html", {
        "classes": GEN_CLASSES,
        "generated_image": generated_image,
    })
