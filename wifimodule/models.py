import uuid

from django.db import models

# Create your models here.
from django.db import models

from django.db import models

from django.db import models

from django.db import models

# Create your models here.
from django.db import models

class TextData(models.Model):
    text_input = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text_input


class DistanceData(models.Model):
    distance = models.FloatField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.distance} cm at {self.timestamp}"


class TemperatureData(models.Model):
    temperature = models.FloatField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.temperature} °C at {self.timestamp}"

class HumidityData(models.Model):
    humidity = models.FloatField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.humidity}% at {self.timestamp}"


class BuzzerData(models.Model):
    status = models.CharField(max_length=10, default="buzzer:0")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.status


# models.py
from django.db import models

class Location(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Location({self.latitude}, {self.longitude})"




from django.db import models

class CameraImage(models.Model):
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='camera_images/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name



class Drawing(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='drawings/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Drawing {self.id} - {self.created_at}"
