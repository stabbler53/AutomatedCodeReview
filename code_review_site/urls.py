"""
URL configuration for code_review_site project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from code_app import views 
from channels.routing import ProtocolTypeRouter, URLRouter
from code_app.consumers import ChatConsumer


application = ProtocolTypeRouter({
    "websocket": URLRouter([
        path("ws/chat/", ChatConsumer.as_asgi()),
    ]),
})

urlpatterns = [
    path('', views.index, name='index'),
    path('service/', views.service, name='service'),
    path('analyze_code/', views.analyze_code, name='code_analysis'),
    path('about/', views.about, name='about'),
    path('chat/', views.chat, name='chat'),
    path('team/', views.team, name='team'),
    path('analyze/', views.analyze_code_view, name='analyze'),
    path('admin/', admin.site.urls),
]
