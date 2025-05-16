"""
URL configuration for kontrolsitesi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from wifimodule import views  # wifimodule uygulamasındaki views dosyasını çağırıyoruz
from wifimodule.views import home50, generate

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Formu gösterecek ana sayfa
    path('get_text/', views.get_text, name='get_text'),  # ESP'nin veri alacağı yol

    path('uzaklik_olcumu/', views.uzaklik_olcumu, name='uzaklik_olcumu'),
    path('add_distance_data/', views.add_distance_data, name='add_distance_data'),

    path('sicaklik_olcumu/', views.sicaklik_olcumu, name='sicaklik_olcumu'),
    path('add_temperature_data/', views.add_temperature_data, name='add_temperature_data'),
    path('nem_olcumu/', views.nem_olcumu, name='nem_olcumu'),
    path('add_humidity_data/', views.add_humidity_data, name='add_humidity_data'),
path('toggle_buzzer/', views.toggle_buzzer, name='toggle_buzzer'),
path('show_map/', views.show_map, name='show_map'),
path('save_location/', views.save_location, name='save_location'),
path('hava_durumu/', views.hava_durumu, name='hava_durumu'),

    path('upload_image/', views.upload_image, name='upload_image'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('canli_goruntu/', views.canli_goruntu, name='canli_goruntu'),

path('predict50/', home50, name='predict50'),
path("generate/", generate, name="generate"),


]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)