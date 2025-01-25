from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

model_names = [
    # "base",
    # "large-v3",
    # "shhossain/whisper-large-bn-v2-ct2",
    "large-v3-turbo-ct2",
    # "turbo"
]


def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(5):
        while True:
            try:
                if selected_model == "large-v3-turbo-ct2":
                    repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
                    local_dir = "faster-whisper-large-v3-turbo-ct2"
                    m = snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")
                    loaded_model = WhisperModel(m, device="cpu", compute_type="int8")
                else:
                    loaded_model = WhisperModel(selected_model, device="cuda", compute_type="int8")
            except (AttributeError, OSError):
                continue

            break

    return selected_model, loaded_model


models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model
