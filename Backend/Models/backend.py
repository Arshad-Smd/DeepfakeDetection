import random
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import librosa
import torch
from typing import List, Tuple
from PIL import Image
import torchvision
import numpy as np
# import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorflow.keras.models import load_model


device = torch.device('cpu')

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = cv2.resize(frame, output_size) 
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)

  return result



def pred(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    
    img = Image.open(image_path)
    

    # print("Model loaded successfully!")
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)


    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    return (class_names[target_image_pred_label] , target_image_pred_probs.max())

def predictFake(path):
    m,_=librosa.load(path,sr=16000)
    max_length=500
    mfccs = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)

    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    model=load_model('C:\\Users\\smdar\\Desktop\\Anokha project\\AudioModel.h5')
    output=model.predict(mfccs.reshape(-1,40,500))
    if output[0][0]>0.5:
        return "fake"
    else:
        return "real"


def save_images(path):
    paths = []
    for i in range(3):
        image_3d = frames_from_video_file(path, 3)[i]
        if image_3d.shape[2] == 4: 
            image_3d = image_3d[:, :, :3]  
        plt.figure(figsize=(1, 1))  
        plt.imshow(image_3d)
        plt.axis('off')
        save_path = f"uploads/image{i}.jpg"
        paths.append(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
        plt.close()  
    return paths



app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}}) 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from collections import Counter

def find_mode(arr):
    counts = Counter(arr)
    max_count = max(counts.values())
    mode = next(key for key, value in counts.items() if value == max_count)
    return mode

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'image' in request.files :
        file = request.files['image']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        with open('pretrained_vit_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        class_names = ['fake','real']
        a = pred(model=loaded_model,
                    image_path=file_path,
                    class_names=class_names)
        os.remove(file_path)
        return jsonify([{'message': 'File uploaded successfully', 'file_path': file_path},a[0]])
    if 'audio' in request.files:
        file = request.files['audio']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        ans=predictFake(file_path)
        print(ans)
        return jsonify([{'message': 'File uploaded successfully'},ans])
    if 'video' in request.files:
        file = request.files['video']
        filename = file.filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        paths = save_images(file_path)
        with open('pretrained_vit_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        class_names = ['fake','real']
        predictions = []
        for i in paths:
            a = pred(model=loaded_model,
                        image_path=i,
                        class_names=class_names)[0]
            os.remove(i)
            predictions.append(a)
        ans=find_mode(predictions)
        
        os.remove(file_path) 
        return jsonify([{'message': 'File uploaded successfully', 'file_path': file_path},ans])
        # return jsonify([{'message': 'File uploaded successfully'},ans])

    


if __name__ == '__main__': 
    app.run(debug=True, port=5000)
