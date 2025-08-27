import queue
from typing import List, Literal, Union
import uuid

import librosa
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from transformers import BitsAndBytesConfig
import torchaudio
from torchaudio.transforms import Resample
from accelerate import (
    load_checkpoint_and_dispatch,
    init_empty_weights,
    infer_auto_device_map,
)

from auto_gptq import AutoGPTQForCausalLM


INPUT_OUTPUT_AUDIO_SAMPLE_RATE = 24000


class AudioData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray
    sample_rate: int


class EndOfResponse:
    pass


class MiniCPMo:
    def __init__(
        self, device: Literal["cpu", "cuda"] = "cuda", model_revision: str = "main"
    ):
        super().__init__()

        self.device = device

        # quant_conf = BitsAndBytesConfig(
        #     load_in_4bit=True,  # Whether to perform 4-bit quantization
        #     load_in_8bit=False,  # Whether to perform 8-bit quantization
        #     bnb_4bit_compute_dtype=torch.float16,  # Compute precision setting
        #     bnb_4bit_quant_storage=torch.uint8,  # Quantized weight storage format
        #     bnb_4bit_quant_type="nf4",  # Quantization format, using normal distribution-based int4 here
        #     bnb_4bit_use_double_quant=True,  # Whether to use double quantization, i.e., quantizing zeropoint and scaling parameters
        #     llm_int8_enable_fp32_cpu_offload=False,  # Whether LLM uses int8, and FP32 is used for parameters stored on the CPU
        #     llm_int8_has_fp16_weight=False,  # Whether mixed precision is enabled
        #     llm_int8_skip_modules=[
        #         "out_proj",
        #         "kv_proj",
        #         "lm_head",
        #         "tts",
        #         "vpm",
        #         "apm",
        #         "resampler",
        #         "audio_avg_pooler",
        #         "audio_projection_layer",
        #     ],  # Modules not to be quantized
        #     llm_int8_threshold=6.0,  # Outlier threshold in the llm.int8() algorithm, used to determine whether to perform quantization
        # )

        # print("Initialized bitsbytes conf")
        # with init_empty_weights():
        #     model = AutoModel.from_pretrained(
        #         "openbmb/MiniCPM-o-2_6",
        #         trust_remote_code=True,
        #         attn_implementation="sdpa",
        #         torch_dtype=torch.bfloat16,
        #         init_audio=False,
        #         # init_tts=False,
        #         init_vision=False,
        #     )
        #     device_map = infer_auto_device_map(
        #         model,
        #         # max_memory={0: "10GB", 1: "10GB"},
        #         no_split_module_classes=[
        #             "SiglipVisionTransformer",
        #             "Qwen2DecoderLayer",
        #         ],
        #     )
        #     # device_id = device_map["llm.model.embed_tokens"]
        #     # device_map[
        #     #     "llm.lm_head"
        #     # ] = device_id  # firtt and last layer should be in same device
        #     # # device_map["vpm"] = device_id
        #     # device_map["tts"] = device_id
        #     # device_map["resampler"] = device_id
        #     # device_id2 = device_map["llm.model.layers.26"]
        #     # device_map["llm.model.layers.8"] = device_id2
        #     # device_map["llm.model.layers.9"] = device_id2
        #     # device_map["llm.model.layers.10"] = device_id2
        #     # device_map["llm.model.layers.11"] = device_id2
        #     # device_map["llm.model.layers.12"] = device_id2
        #     # device_map["llm.model.layers.13"] = device_id2
        #     # device_map["llm.model.layers.14"] = device_id2
        #     # device_map["llm.model.layers.15"] = device_id2
        #     # device_map["llm.model.layers.16"] = device_id2
        #     # print(device_map)

        # model = load_checkpoint_and_dispatch(
        #     model, "openbmb/MiniCPM-o-2_6", dtype=torch.bfloat16, device_map=device_map
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "openbmb/MiniCPM-o-2_6", trust_remote_code=True
        # )
        # model.eval()

        # self.model = (
        #     AutoModel.from_pretrained(
        #         "openbmb/MiniCPM-o-2_6",
        #         trust_remote_code=True,
        #         attn_implementation="sdpa",
        #         # attn_implementation="flash_attention_2",
        #         torch_dtype=torch.bfloat16,
        #         revision=model_revision,
        #         # quantization_config=quant_conf,
        #         low_cpu_mem_usage=True,
        #         device_map=device,
        #         init_vision=False,
        #         init_audio=False,
        #     ).eval()
        #     # .to(device)
        # )
        print("model initialize")
        # model = AutoGPTQForCausalLM.from_quantized(
        #     "openbmb/MiniCPM-o-2_6-int4",
        #     torch_dtype=torch.bfloat16,
        #     device=device,
        #     trust_remote_code=True,
        #     disable_exllama=True,
        #     disable_exllamav2=True,
        # )
        self.model = AutoGPTQForCausalLM.from_quantized(
            "openbmb/MiniCPM-o-2_6-int4",
            torch_dtype=torch.bfloat16,
            device=self.device + ":0",
            trust_remote_code=True,
            disable_exllama=True,
            disable_exllamav2=True,
            init_vision=False,
            init_audio=False,
        )

        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=True,
        )

        # self._tokenizer = AutoTokenizer.from_pretrained(
        #     "openbmb/MiniCPM-o-2_6", trust_remote_code=True, revision=model_revision
        # )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-2_6-int4",
            trust_remote_code=True,  # , revision=model_revision
        )
        print("tokenizer initialize")

        # self._tokenizer = torch.compile(
        #     self._tokenizer, mode="max-autotune", fullgraph=True
        # )

        self.resampler = Resample(new_freq=24000).to(self.device)

        if device == "cuda":
            self.init_tts()

        self._generate_audio = True
        print("✅ MiniCPMo initialized")

    def init_tts(self):
        self.model.init_tts()
        self.model.tts.bfloat16()

    def _prefill_audio(
        self,
        audio_arrays: List[np.ndarray],
    ):
        audio_samples = np.concatenate(audio_arrays)
        print(f"prefilling audio with {audio_samples.shape} samples")

        chunk_size = INPUT_OUTPUT_AUDIO_SAMPLE_RATE
        for chunk_start in range(0, len(audio_samples), chunk_size):
            chunk = audio_samples[chunk_start : chunk_start + chunk_size]

            msgs = [{"role": "user", "content": [chunk]}]

            self.model.streaming_prefill(
                session_id=self.session_id,
                msgs=msgs,
                tokenizer=self._tokenizer,
            )

    def _prefill(self, data: List[str | AudioData]):
        # try:
        #     audio_arrays = []
        #     for prefill_data in data:
        #         if isinstance(prefill_data, str):
        #             text = prefill_data
        #             audio = None
        #         elif isinstance(prefill_data, AudioData):
        #             text = None
        #             audio = prefill_data.array
        #         else:
        #             raise ValueError(
        #                 f"._prefill(): prefill_data must be a string or AudioData"
        #             )

        #         if text:
        #             self.model.streaming_prefill(
        #                 session_id=self.session_id,
        #                 msgs=[{"role": "user", "content": [text]}],
        #                 tokenizer=self._tokenizer,
        #             )

        #         if audio is not None:
        #             audio_tensor = torch.from_numpy(audio).to(self.device).float()
        #             resampled_audio = self.resampler(audio_tensor).cpu().numpy()
        #             # resampled_audio = librosa.resample(audio, audio.sample_rate, 24000)

        #             self._prefill_audio(
        #                 audio_arrays=[resampled_audio],
        #             )

        # except Exception as e:
        #     print(f"_prefill() error: {e}")
        #     raise e

        try:
            audio_queue = []
            for prefill_data in data:
                if isinstance(prefill_data, str):
                    if audio_queue:
                        self._prefill_audio(audio_arrays=audio_queue)
                        audio_queue = []
                    # with torch.amp.autocast(self.device):
                    self.model.streaming_prefill(
                        session_id=self.session_id,
                        msgs=[{"role": "user", "content": [prefill_data]}],
                        tokenizer=self._tokenizer,
                    )
                elif isinstance(prefill_data, AudioData):
                    audio_tensor = (
                        torch.from_numpy(prefill_data.array).to(self.device).float()
                    )
                    resampled_audio = self.resampler(audio_tensor).cpu().numpy()
                    audio_queue.append(resampled_audio)
                else:
                    raise ValueError(
                        f"._prefill(): prefill_data must be a string or AudioData"
                    )

        except Exception as e:
            print(f"_prefill() error: {e}")
            raise e

    def run_inference(self, prefill_data: List[str | AudioData]):
        print("MiniCPMo _run_inference() function called")

        try:
            for data in prefill_data:
                self.session_id = str(uuid.uuid4())
                self._prefill(data=[data])

                # with torch.amp.autocast(self.device):
                response_generator = self.model.streaming_generate(
                    session_id=self.session_id,
                    tokenizer=self._tokenizer,
                    temperature=0.1,
                    generate_audio=self._generate_audio,
                    use_cache=True,  # check if model has KV cache
                    sampling=True,  # doc says faster inferencing
                )

                for response in response_generator:
                    audio = None
                    sample_rate = INPUT_OUTPUT_AUDIO_SAMPLE_RATE
                    text = None

                    # extract audio from response
                    if hasattr(response, "audio_wav"):
                        has_audio = True
                        sample_rate = getattr(
                            response, "sampling_rate", INPUT_OUTPUT_AUDIO_SAMPLE_RATE
                        )
                        audio = response.audio_wav.cpu().detach().numpy()

                    # check for text
                    if isinstance(response, dict):
                        text = response.get("text")
                    elif hasattr(response, "text"):
                        text = response.text

                    # put audio in output queue
                    if audio is not None:
                        audio_data = AudioData(
                            array=audio,
                            sample_rate=sample_rate,
                        )

                        yield audio_data

                    # put text in output queue
                    if isinstance(text, str) and text:
                        has_text = True
                        yield text

                yield EndOfResponse()

        except Exception as e:
            print(f"_run_inference() error: {e}")
            yield None
