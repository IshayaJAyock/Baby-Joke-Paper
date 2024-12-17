import os
import numpy as np
import cv2
import librosa
import dlib
from moviepy.editor import VideoFileClip
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from config import VIDEO_FOLDER, N_FRAMES, SHAPE_PREDICTOR_PATH

def sample_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * i / n_frames))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return np.array(frames)

def extract_visual_features(frames):
    base_model = VGG16(weights='imagenet', include_top=False)
    features = base_model.predict(preprocess_input(frames))
    return np.mean(features, axis=0).flatten()[:512]

def extract_facial_features(frames):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    facial_features = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            shape = predictor(gray, face)
            facial_features.append(np.array([(p.x, p.y) for p in shape.parts()]).flatten())
    return np.mean(facial_features, axis=0) if facial_features else np.zeros(136)

def extract_audio_features(video_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            print(f"No audio found in {video_path}")
            return np.zeros(54)
        audio_array = audio.to_soundarray()
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        mfcc = librosa.feature.mfcc(y=audio_array, sr=audio.fps, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=audio.fps)
        chroma = librosa.feature.chroma_stft(y=audio_array, sr=audio.fps)
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        video.close()
        return features[:54]
    except Exception as e:
        print(f"Error extracting audio features from {video_path}: {str(e)}")
        return np.zeros(54)

def extract_motion_features(frames):
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    motion_features = []
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_features.append(np.mean(magnitude))
        prev_gray = gray
    return np.array(motion_features)

def extract_interaction_features(frames):
    detector = dlib.get_frontal_face_detector()
    interaction_features = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 2:
            face1, face2 = faces
            distance = np.sqrt((face1.center().x - face2.center().x)**2 + 
                               (face1.center().y - face2.center().y)**2)
            interaction_features.append(distance)
    return np.mean(interaction_features) if interaction_features else 0

def extract_temporal_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    cap.release()
    return np.array([frame_count, fps, duration])[:3]

def extract_all_features(video_file):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    try:
        frames = sample_frames(video_path, N_FRAMES)
        visual_feat = extract_visual_features(frames)
        facial_feat = extract_facial_features(frames)
        audio_feat = extract_audio_features(video_path)
        motion_feat = extract_motion_features(frames)
        interaction_feat = extract_interaction_features(frames)
        temporal_feat = extract_temporal_features(video_path)
        
        combined_feat = np.concatenate([
            visual_feat, facial_feat, audio_feat, 
            motion_feat, [interaction_feat], temporal_feat
        ])
        return combined_feat
    except Exception as e:
        print(f"Error processing {video_file}: {str(e)}")
        return None