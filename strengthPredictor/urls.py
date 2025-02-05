from django.urls import path
from .views import predict_strength

urlpatterns = [
    path('', predict_strength, name='predict_strength'),
]
