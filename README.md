# Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Setup Modal:

```bash
modal setup
```

# Usage

1. Get your huggingface token: https://huggingface.co/settings/tokens

2. Add your huggingface token at the top of inference.py file:

```python
HF_TOKEN = "<your-huggingface-token>"
```

3. Run inference:

```bash
modal run inference.py
```

# Introduction

This took me sometime but I got an opportunity to learn the internal workings of the model better. I have made a couple of observations (most of them are tested and commented out in the previous commit). 

# Possible Solutions

1. Load Models directly to GPU
2. Quantization techniques: BitsAndBytes, torch.compile (Static Graph), torch.amp.autocast
3. Inference optimization through llama.cpp, onnx
4. Acceleration using accelerate
5. Prevent Vision part initialization
6. Attention implementation = sdpa/flash_attention_2
7. Batch inference

# Challenges

1. Dependency issue. Original code's python and cuda mismatch. torch, accelerate, and bitsandbytes mismatch
2. Quantizing model through bitsandbytes is not possible without large changes because of tensors moving from CPU to GPU memory internally in the code. 
3. LLama cpp gguf model is fast but works only with the LLM layer. Did not explore porting it to audio.
4. Did not see performance gains with torch.amp.autocast. torch.bfloat16 loads model in float16 
5. int4 bit quantized model available with AutoGPTQ. AutoGPTQ fails with original version of the container. 
6. Modal is inconsistant with TTFB and RTF. 

# Implemented solution

1. Load all models directly on gpu.
2. Use CUDA for resampling (effective for large amounts of data)
3. Installed AutoGPTQ with the right deps for int4 quantized model
4. Batch inference
5. Reduced Sampling rate multiplier (Chunk size). Slight improvement in performance. 
6. Removed Vision and Audio init during model loading

# Observations

1. Using original MiniCPM-O with quantized tokenizer gives good performance improvements.

# Things to explore

1. Quantize with BitsAndBytes - 8 bit might lead to better quality
2. Onnx for stage 1 (LLM infer) with streaming TTS
3. Reduce the output token size of the first inference pass (only) 
4. Output Caching.
5. CPU reconstruction of MelSpectrogram (griffin_lim)
6. Tokenizer pruning. Model Distillation

# Results

The results are in the output/optimized folder.
Average TTFB and RTF
Time to first byte: 1.2783586621666665
Realtime Factor: 0.8420800145118292

# Instructions to Run

Add your huggingface token in a $PWD/.env or your ~/.bashrc

You can use either virtualenv or uv

``` bash
uv venv
uv sync
```

``` bash
source .venv/bin/activate
```

Run 

``` bash
modal run inference.py
```

