from onnxruntime.quantization import quantize_dynamic, QuantType
import os

src = "combined_640.onnx"
dst = "combined_640_int8.onnx"
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)

a, b = os.path.getsize(src)/1e6, os.path.getsize(dst)/1e6
print(f"{src}: {a:.1f} MB")
print(f"{dst}: {b:.1f} MB   ({a/b:.2f}x smaller)")
