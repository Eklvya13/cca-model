from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLARITY_MAP = {"happy": "positive", "angry": "negative", "fear": "negative", "sad": "negative", "neutral": "neutral", "disgust": "negative", "surprise": "neutral"}

class EmotionDetectorLocal:
    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        )
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        ).to(DEVICE)
        self.model.eval()

    def predict_batch(self, audio_slices, sampling_rate):
        inputs = self.feature_extractor(
            audio_slices,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=250000
        )
        input_values = inputs["input_values"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(audio_slices)):
            confidence, predicted_label = torch.max(probs[i], dim=-1)
            emotion = self.model.config.id2label[predicted_label.item()]
            emotion_score = round(confidence.item(), 4)
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
