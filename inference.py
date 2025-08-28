from pathlib import Path
from datetime import datetime
import os

import modal
import numpy as np
import soundfile as sf
from typing import List


HF_TOKEN = os.getenv("HF_TOKEN")


MODEL_REVISION = "9da79acdd8906c7007242cbd09ed014d265d281a"


app = modal.App(name="minicpm-inference-engine2")


minicpm_inference_engine_image = (
    # install MiniCPM-o dependencies
    modal.Image.from_registry(
        # f"nvidia/cuda:12.6.0-devel-ubuntu22.04",
        # f"pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel",
        f"pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "cmake", "clang")
    .run_commands("pip install --upgrade pip")
    # Install flash-attn dependencies
    .pip_install(  # required to build flash-attn
        "ninja==1.11.1.3",
        "packaging==24.2",
        "wheel",
        # "torch==2.7.1",
        # "torchaudio==2.7.1",
        # "torchvision==0.22.1",
        "torchaudio==2.3.1",
        "torchvision==0.18.1",
        "huggingface_hub[hf_transfer]==0.30.1",
        "huggingface_hub[hf_xet]",
        "transformers==4.44.2",
        "onnxruntime==1.20.1",
        "scipy==1.15.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "Pillow==10.1.0",
        "sentencepiece==0.2.0",
        "vector-quantize-pytorch==1.18.5",
        "vocos==0.1.0",
        "timm==0.9.10",
        "soundfile==0.12.1",
        "librosa==0.9.0",
        "sphn==0.1.4",
        "aiofiles==23.2.1",
        "decord",
        "moviepy",
        "pydantic",
        gpu="A10G",
    )
    .run_commands(
        "pip install --upgrade flash_attn==2.5.8", gpu="A10G"
    )  # add flash-attn
    .pip_install("accelerate==1.2.1", gpu="A10G")
    .pip_install("bitsandbytes==0.45.3", gpu="A10G")
    # .pip_install("bitsandbytes==0.47.0", gpu="A10G")
    .run_commands('python -c "import torch; print(torch.version.cuda)"')
    # .run_commands("accelerate config default")
    # .run_commands("accelerate env")
    .run_commands(
        "git clone https://github.com/RanchiZhao/AutoGPTQ.git",
        "cd AutoGPTQ && git checkout minicpmo",
        "cd AutoGPTQ && python setup.py clean",
        "cd AutoGPTQ && pip install -vvv --no-build-isolation .",
        gpu="A10G",
    )
    # .run_commands(
    #     "git clone https://github.com/huggingface/accelerate",
    #     "cd accelerate && git checkout v1.3.0",
    #     "cd accelerate && pip install -e .",
    #     gpu="A10G",
    # )
    # .run_commands(
    #     "git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git &&  cd bitsandbytes/ && git checkout 0.47.0",
    #     "cd bitsandbytes/ && cmake -DCOMPUTE_BACKEND=cuda -S .",
    #     "cd bitsandbytes/ && make -j$(nproc)",
    #     "cd bitsandbytes/ && pip install .",
    #     gpu="A10G",
    # )
    .run_commands("python -m bitsandbytes", gpu="A10G")
    .pip_install("gekko")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",
            "HF_TOKEN": HF_TOKEN,
        }
    )  # and enable it
    .add_local_python_source("minicpmo")
)

# onnx_model_path = "/root/minicpm_core_encoder.onnx"
# onnx_model_mount = modal.Mount.from_local_file(
#     "minicpm_core_encoder.onnx", remote_path=onnx_model_path
# )

with minicpm_inference_engine_image.imports():
    import time
    from minicpmo import MiniCPMo, AudioData, EndOfResponse
    import numpy as np


MODAL_GPU = "A10G"


@app.cls(
    cpu=2,
    memory=5000,
    gpu=MODAL_GPU,
    image=minicpm_inference_engine_image,
    min_containers=1,
    timeout=15 * 60,
    volumes={
        "/cache": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
)
class MinicpmInferenceEngine:
    @modal.enter()
    def load(self):
        self.model = MiniCPMo(device="cuda", model_revision=MODEL_REVISION)
        self.model.init_tts()

    @modal.method()
    def run(self, text: str):
        audio_data = []
        start_time = time.perf_counter()
        time_to_first_byte = None

        # Generator for a single text
        item_generator = self.model.run_inference(text)

        for item in item_generator:
            if isinstance(item, EndOfResponse):
                break

            if isinstance(item, AudioData):
                if time_to_first_byte is None:
                    time_to_first_byte = time.perf_counter() - start_time
                audio_data.append(item.array)

        total_time = time.perf_counter() - start_time
        full_audio = np.concatenate(audio_data) if audio_data else np.array([])

        return {
            "time_to_first_byte": time_to_first_byte,
            "total_time": total_time,
            "audio_array": full_audio,
            "sample_rate": 24000,
            "text": text,
        }


@app.local_entrypoint()
def main():
    INF_TYPE = "optimized"  # original/optimized
    engine = MinicpmInferenceEngine()
    print("start")
    # Warmup
    engine.run.remote("Hi, how are you?")

    results = []
    texts = [
        "I'm fine, thank you!",
        "What's your name?",
        "My name is John Doe",
        "What's your favorite color?",
        "My favorite color is blue",
        "What's your favorite food?",
    ]
    for result in engine.run.map(texts):
        results.append(result)

    PARENT_DIR = Path(__file__).parent

    for result in results:
        sf.write(
            PARENT_DIR / "outputs" / INF_TYPE / f"{result['text']}.wav",
            result["audio_array"],
            result["sample_rate"],
        )
        print(
            f"Wrote {result['text']}.wav to {PARENT_DIR / 'outputs' / INF_TYPE / result['text']}.wav"
        )

    ttfb_results = [r['time_to_first_byte'] for r in results if r['time_to_first_byte'] is not None]
    if ttfb_results:
        print(f"Time to first byte: {np.mean(ttfb_results)}")

    rtf_results = [r['total_time'] / (len(r['audio_array']) / r['sample_rate']) for r in results if len(r['audio_array']) > 0]
    if rtf_results:
        print(f"Realtime Factor: {np.mean(rtf_results)}")

    now = datetime.now()
    result_fn = (
        PARENT_DIR
        / "outputs"
        / INF_TYPE
        / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_results.txt"
    )
    with open(result_fn, "w+") as f:
        f.write(f"Results from run at {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for i, result in enumerate(results):
            if len(result['audio_array']) > 0:
                rtf = result["total_time"] / (
                    len(result["audio_array"]) / result["sample_rate"]
                )
            else:
                rtf = 0
            
            f.write(f"Result {i+1}: {result['text']}\n")
            ttfb = result['time_to_first_byte']
            if ttfb is not None:
                f.write(f"-Time to first byte: {ttfb:.4f}s\n")
            else:
                f.write(f"-Time to first byte: N/A\n")
            f.write(f"-Realtime Factor: {rtf:.4f}x\n")
            f.write(f"-Total time: {result['total_time']:.4f}s\n")
            f.write(
                f"-Audio length: {len(result['audio_array']) / result['sample_rate']:.4f}s\n"
            )
