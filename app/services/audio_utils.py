import librosa
import numpy as np

def slice_audio(audio, sr, slice_duration, rms_threshold):
    slice_samples = int(slice_duration * sr)
    total_samples = len(audio)
    slices = []

    for start in range(0, total_samples, slice_samples):
        end = min(start + slice_samples, total_samples)
        audio_slice = audio[start:end]

        rms = np.sqrt(np.mean(audio_slice ** 2))
        is_silent = rms < rms_threshold

        if not is_silent:
            slices.append({
                "start_time": round(start / sr, 2),
                "end_time": round(end / sr, 2),
                "audio_slice": audio_slice
            })

    return slices
