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
        # "transformers==4.45.1",
        "transformers==4.44.2",
        "onnxruntime==1.20.1",
        "scipy==1.15.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "Pillow==10.1.0",
        "sentencepiece==0.2.0",
        "vector-quantize-pytorch==1.18.5",
        "vocos==0.1.0",
        # "accelerate",
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
        "pip install --upgrade flash_attn==2.8.0.post2", gpu="A10G"
    )  # add flash-attn
    .pip_install("accelerate==1.2.1", gpu="A10G")
    .pip_install("bitsandbytes==0.45.3", gpu="A10G")
    # .pip_install("bitsandbytes==0.47.0", gpu="A10G")
    .run_commands(
        "python -c \"import torch; import transformers; import accelerate; import bitsandbytes; print(f'\\n==== Library Versions ===='); print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'Accelerate: {accelerate.__version__}'); print(f'BitsAndBytes: {bitsandbytes.__version__}'); print('\\n==== CUDA Info ===='); print(f'CUDA is available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print('\\n==== BitsAndBytes Test Output ====');\"",
        gpu="A10G",
    )
    .run_commands('python -c "import torch; print(torch.version.cuda)"')
    # Install general AI dependencies
    # .pip_install("accelerate>=0.30.0", gpu="A10G")
    # .pip_install("bitsandbytes", gpu="A10G")
    # .run_commands("accelerate config default")
    # .run_commands("accelerate env")
    # .run_commands("python -m bitsandbytes")
    .run_commands(
        "git clone https://github.com/RanchiZhao/AutoGPTQ.git",
        "cd AutoGPTQ && git checkout minicpmo",
        "cd AutoGPTQ && python setup.py clean",
        "cd AutoGPTQ && pip install -vvv --no-build-isolation .",
        gpu="A10G",
    )
    # .run_commands("cd ..")
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
    def run(self, texts: List[str]):
        all_results = []
        audio_data = []
        start_time = time.perf_counter()
        time_to_first_byte = None
        text_index = 0

        # A single call to the generator for the whole batch
        item_generator = self.model.run_inference(texts)

        for item in item_generator:
            if isinstance(item, EndOfResponse):
                # We've reached the end of a response for one text
                total_time = time.perf_counter() - start_time
                full_audio = np.concatenate(audio_data) if audio_data else np.array([])

                all_results.append(
                    {
                        "time_to_first_byte": time_to_first_byte,
                        "total_time": total_time,
                        "audio_array": full_audio,
                        "sample_rate": 24000,
                        "text": texts[text_index],
                    }
                )

                # Reset for the next text in the batch
                audio_data = []
                start_time = time.perf_counter()
                time_to_first_byte = None
                text_index += 1
                continue

            # Process audio and text items as before
            if isinstance(item, AudioData):
                if time_to_first_byte is None:
                    time_to_first_byte = time.perf_counter() - start_time
                audio_data.append(item.array)

        return all_results

        # audio_data = []
        # start_time = time.perf_counter()
        # time_to_first_byte = None
        # total_time = None
        # for item in self.model.run_inference([text]):
        #     if item is None:
        #         break
        #     if isinstance(item, str):
        #         print(f"Got text from MiniCPM: {text}")
        #     if isinstance(item, AudioData):
        #         assert item.sample_rate == 24000

        #         if time_to_first_byte is None:
        #             time_to_first_byte = time.perf_counter() - start_time

        #         audio_data.append(item.array)

        # total_time = time.perf_counter() - start_time

        # if len(audio_data) == 0:
        #     raise ValueError("No audio data received")

        # full_audio = np.concatenate(audio_data)

        # return {
        #     "time_to_first_byte": time_to_first_byte,
        #     "total_time": total_time,
        #     "audio_array": full_audio,
        #     "sample_rate": 24000,
        #     "text": text,
        # }


@app.local_entrypoint()
def main():
    INF_TYPE = "optimized"  # original/optimized
    engine = MinicpmInferenceEngine()
    print("start")
    # Warmup
    result = engine.run.remote(["Hi, how are you?"])

    results = []
    texts = [
        "I'm fine, thank you!",
        "What's your name?",
        "My name is John Doe",
        "What's your favorite color?",
        "My favorite color is blue",
        "What's your favorite food?",
    ]
    results = engine.run.remote(texts)
    # for text in [
    #     "I'm fine, thank you!",
    #     "What's your name?",
    #     "My name is John Doe",
    #     "What's your favorite color?",
    #     "My favorite color is blue",
    #     "What's your favorite food?",
    # ]:
    #     result = engine.run.remote(text)
    #     results.append(result)

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

    print(
        f"Time to first byte: {np.mean([result['time_to_first_byte'] for result in results])}"
    )
    print(
        f"Realtime Factor: {np.mean([result['total_time'] / (len(result['audio_array']) / result['sample_rate']) for result in results])}"
    )
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
            rtf = result["total_time"] / (
                len(result["audio_array"]) / result["sample_rate"]
            )
            f.write(f"Result {i+1}: {result['text']}\n")
            f.write(f"-Time to first byte: {result['time_to_first_byte']:.4f}s\n")
            f.write(f"-Realtime Factor: {rtf:.4f}x\n")
            f.write(f"-Total time: {result['total_time']:.4f}s\n")
            f.write(
                f"-Audio length: {len(result['audio_array']) / result['sample_rate']:.4f}s\n"
            )
