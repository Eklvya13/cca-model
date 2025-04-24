from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import os

# Make sure this path points to the locally downloaded model dir in your Docker image
LOCAL_MODEL_PATH = "models/whisper-large-v3-emotion"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated emotion label map — please double-check the exact label set from the config
POLARITY_MAP = {
    "happy": "positive",
    "angry": "negative",
    "calm": "positive",
    "sad": "negative",
    "neutral": "neutral",
    "disgust": "negative",
    "surprised": "positive",
    "fearful": "negative"
}

class EmotionDetectorLocal:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForAudioClassification.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print(DEVICE)

    def preprocess_audio(self, audio_array, sampling_rate=16000, max_duration=30.0):
        max_length = int(self.feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = torch.nn.functional.pad(
                torch.tensor(audio_array), (0, max_length - len(audio_array))
            )

        inputs = self.feature_extractor(
            audio_array.numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def predict_batch(self, audio_slices, sampling_rate):
        results = []
        for audio_slice in audio_slices:
            inputs = self.preprocess_audio(audio_slice, sampling_rate)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = self.id2label[predicted_id]
            
            emotion = predicted_label
            emotion_score = round(torch.nn.functional.softmax(logits, dim=-1)[0, predicted_id].item(), 4)
            absolute_emotion = POLARITY_MAP.get(emotion, "neutral")
            
            results.append({
                "real_emotion": emotion,
                "absolute_emotion": absolute_emotion,
                "emotion_score": emotion_score,
            })
        
        return results

    def predict_in_mini_batches(self, audio_slices, batch_size=4, sampling_rate=16000):
        results = []
        for i in range(0, len(audio_slices), batch_size):
            batch = audio_slices[i:i + batch_size]
            results.extend(self.predict_batch(batch, sampling_rate))
        return results
