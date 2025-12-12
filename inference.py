import torch
import torch.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import time
from src.model import ResEmoteNet

def optimize_for_edge():
    # 1. Load Trained PyTorch Model
    print("Loading PyTorch model...")
    model = ResEmoteNet(num_classes=7)
    model.load_state_dict(torch.load("resemotenet.pth"))
    model.eval()

    # 2. Export to ONNX (Standard Format)
    dummy_input = torch.randn(1, 1, 48, 48)
    onnx_path = "resemotenet.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model exported to {onnx_path}")

    # 3. Apply Quantization (Float32 -> Int8)
    # Resume Point: "Optimized model using INT8 quantization for edge devices"
    quantized_path = "resemotenet_int8.onnx"
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print(f"Quantized model saved to {quantized_path}")

    # 4. Latency Benchmark (The "Proof")
    print("\n--- Benchmarking Inference Speed ---")
    
    # Standard ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    # Quantized ONNX Runtime
    q_session = ort.InferenceSession(quantized_path)
    
    input_data = np.random.randn(1, 1, 48, 48).astype(np.float32)
    
    # Warmup
    for _ in range(10): session.run(None, {'input': input_data})

    # Measure Standard
    start = time.time()
    for _ in range(100): session.run(None, {'input': input_data})
    avg_latency = (time.time() - start) / 100 * 1000
    print(f"Standard ONNX Latency: {avg_latency:.2f} ms")

    # Measure Quantized
    start = time.time()
    for _ in range(100): q_session.run(None, {'input': input_data})
    avg_q_latency = (time.time() - start) / 100 * 1000
    print(f"Quantized INT8 Latency: {avg_q_latency:.2f} ms")
    
    print(f"Speedup: {avg_latency / avg_q_latency:.2f}x faster on CPU")

if __name__ == "__main__":
    optimize_for_edge()