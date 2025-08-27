import torch
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort
import os

# Make sure to set your HF_TOKEN if needed
# HF_TOKEN = os.getenv("HF_TOKEN")


def export_core_model_to_onnx():
    """
    This script exports the core transformer model (e.g., Qwen2) from inside
    the complex MiniCPM-o wrapper. This avoids the non-exportable custom code.
    """
    print("Loading PyTorch model...")
    # Load the base MiniCPM-o model, which acts as a wrapper
    wrapper_model = (
        AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-2_6",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        .eval()
        .to("cuda")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-o-2_6", trust_remote_code=True
    )

    # --- 1. Access the Core Model and Create Dummy Inputs ---

    # Based on inspecting the object, the core transformer is in the .llm attribute.
    core_model = wrapper_model.llm
    print(f"Identified core model of type: {type(core_model)}")

    text = "This is a sample sentence for ONNX export."
    dummy_inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Standard transformers expect input_ids and attention_mask as separate arguments.
    # We pass this as a tuple of arguments to the exporter.
    dummy_args = (dummy_inputs.input_ids, dummy_inputs.attention_mask)

    print(f"Dummy input shape: {dummy_inputs.input_ids.shape}")

    # --- 2. Export the Core Model to ONNX ---
    output_path = "minicpm_core_encoder.onnx"
    print(f"Exporting core model to {output_path}...")

    torch.onnx.export(
        core_model,  # Export the simpler, core model
        args=dummy_args,  # Pass input_ids and attention_mask
        f=output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    print(f"\nSuccessfully exported to {output_path}")

    # --- 3. Verify the ONNX Model ---
    print("Verifying the ONNX model with ONNX Runtime...")
    try:
        ort_session = ort.InferenceSession(
            output_path, providers=["CUDAExecutionProvider"]
        )

        ort_inputs = {
            "input_ids": dummy_inputs.input_ids.cpu().numpy(),
            "attention_mask": dummy_inputs.attention_mask.cpu().numpy(),
        }
        ort_outs = ort_session.run(None, ort_inputs)

        print("ONNX model verified successfully!")
        print("Output shape from ONNX Runtime:", ort_outs[0].shape)
    except Exception as e:
        print(f"An error occurred during ONNX verification: {e}")


if __name__ == "__main__":
    export_core_model_to_onnx()