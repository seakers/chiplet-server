"""
URL configuration for my_backend project.

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
# my_backend/urls.py
from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def scatterplot(request):
    # Generate random data
    x = np.random.rand(50)
    y = np.random.rand(50)

    # Create a scatter plot
    plt.figure()
    plt.scatter(x, y)
    plt.title("Random Scatterplot")

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(image_png).decode('utf-8')

    # Return the image as an HTML response
    html = f'<img src="data:image/png;base64,{image_base64}" />'
    return HttpResponse(html)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('', scatterplot),
]
