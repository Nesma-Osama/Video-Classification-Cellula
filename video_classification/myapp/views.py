from django.shortcuts import render
from .forms import VideoUploadForm
from .utils import predict_video
import tensorflow as tf
# import tensorflow_hub as hub
from django.conf import settings
import os

# Load model globally
# custom_objects = {'KerasLayer': hub.KerasLayer}
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp/models/model.h5')
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
model = tf.keras.models.load_model(MODEL_PATH)

def home(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = request.FILES['video']
            video_path = os.path.join(settings.MEDIA_ROOT, video.name)
            with open(video_path, 'wb+') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)
            label, confidence = predict_video(model, video_path)
            os.remove(video_path)
            return render(request, 'myapp/result.html', {'label': label, 'confidence': f'{confidence:.2%}'})
            # return render(request, 'myapp/home.html', {'form': form})
    else:
        form = VideoUploadForm()
    return render(request, 'myapp/home.html', {'form': form})