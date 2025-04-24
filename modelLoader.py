from huggingface_hub import snapshot_download
import os

def download_model_locally():
    # Set your destination directory
    local_dir = "./models/whisper-large-v3-emotion"
    os.makedirs(local_dir, exist_ok=True)

    # Download full snapshot from Hugging Face
    snapshot_download(
        repo_id="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Ensures real files are copied (better for Docker)
    )

    print(f"Model downloaded and saved to: {local_dir}")

if __name__ == "__main__":
    download_model_locally()
