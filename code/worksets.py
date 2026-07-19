#!/usr/bin/env python3
"""
worksets.py — Compute per-layer working sets from ONNX model graphs.

For each compute layer (Conv, Gemm, MatMul) in the model, computes the total
memory footprint of its inputs + outputs (the "working set" that must be resident
during that layer's execution). Compares this against L3 cache size to estimate
how many layers must re-stream from DRAM.

Usage:
    python3 worksets.py                       # processes all yolo11*.onnx in cwd
    python3 worksets.py --models yolo11s.onnx yolo11m.onnx
    python3 worksets.py --l3-kb 2048          # default: 2048 KB (2 MB)

Outputs worksets.json with per-model summary.
"""

import argparse
import glob
import json
import os
import sys

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("ERROR: onnx package required.  pip install onnx")
    sys.exit(1)

import numpy as np


DTYPE_BYTES = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    4: 2,   # UINT16
    5: 2,   # INT16
    6: 4,   # INT32
    7: 8,   # INT64
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    16: 2,  # BFLOAT16
}

COMPUTE_OPS = {'Conv', 'Gemm', 'MatMul', 'ConvTranspose'}


def tensor_size_bytes(shape, dtype_enum=1):
    """Compute tensor size in bytes from shape and ONNX dtype enum."""
    if not shape or 0 in shape:
        return 0
    elem = 1
    for d in shape:
        if d < 0:
            d = 1  # dynamic dim, assume 1
        elem *= d
    return elem * DTYPE_BYTES.get(dtype_enum, 4)


def get_shape_map(model):
    """Build a map from tensor name to (shape, dtype) using shape inference."""
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass  # proceed without inference

    shapes = {}

    # From graph inputs
    for vi in model.graph.input:
        s = [d.dim_value if d.dim_value > 0 else 1
             for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = (s, vi.type.tensor_type.elem_type)

    # From initializers (weights)
    for init in model.graph.initializer:
        shapes[init.name] = (list(init.dims), init.data_type)

    # From value_info (intermediate tensors after shape inference)
    for vi in model.graph.value_info:
        s = [d.dim_value if d.dim_value > 0 else 1
             for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = (s, vi.type.tensor_type.elem_type)

    # From graph outputs
    for vi in model.graph.output:
        s = [d.dim_value if d.dim_value > 0 else 1
             for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = (s, vi.type.tensor_type.elem_type)

    return shapes


def analyze_model(path, l3_bytes, measured_mb=None):
    """Analyze a single ONNX model and return summary dict."""
    model = onnx.load(path)
    shapes = get_shape_map(model)

    # Weight size
    weight_bytes = sum(
        tensor_size_bytes(list(init.dims), init.data_type)
        for init in model.graph.initializer
    )

    # Initializer names (to distinguish weights from activations)
    init_names = {init.name for init in model.graph.initializer}

    layers = []
    for node in model.graph.node:
        if node.op_type not in COMPUTE_OPS:
            continue
        ws = 0
        for name in list(node.input) + list(node.output):
            if name in shapes:
                s, dt = shapes[name]
                ws += tensor_size_bytes(s, dt)
        layers.append({
            'op': node.op_type,
            'name': node.name or f"{node.op_type}_{len(layers)}",
            'workset_bytes': ws,
        })

    total_ws = sum(l['workset_bytes'] for l in layers)
    over_l3 = sum(1 for l in layers if l['workset_bytes'] > l3_bytes)

    result = {
        'weight_MB': round(weight_bytes / 1e6, 1),
        'total_workset_MB': round(total_ws / 1e6, 1),
        'n_compute_layers': len(layers),
        'layers_over_L3': over_l3,
        'frac_layers_over_L3': round(over_l3 / len(layers), 3) if layers else 0,
    }

    if measured_mb is not None:
        result['measured_MB_per_img'] = measured_mb
        result['dram_amplification'] = round(measured_mb / (weight_bytes / 1e6), 1)
        result['pred_over_meas'] = round(total_ws / 1e6 / measured_mb, 3)

    return result


def main():
    ap = argparse.ArgumentParser(description='ONNX working-set analyzer')
    ap.add_argument('--models', nargs='+', default=None,
                    help='ONNX files to analyze (default: all yolo11*.onnx in cwd)')
    ap.add_argument('--l3-kb', type=int, default=2048,
                    help='L3 cache size in KB (default: 2048 = 2 MB)')
    ap.add_argument('--output', default='worksets.json',
                    help='Output JSON file (default: worksets.json)')
    ap.add_argument('--measured', default=None,
                    help='JSON mapping model name to measured MB/img, e.g. '
                         '\'{"yolo11n":397,"yolo11s":1002}\'')
    a = ap.parse_args()

    l3_bytes = a.l3_kb * 1024

    if a.models:
        files = a.models
    else:
        files = sorted(glob.glob('yolo11*.onnx'))
        # Exclude resolution-specific and int8 variants
        files = [f for f in files if '_1088' not in f and '_1152' not in f
                 and 'int8' not in f and '_640' not in f]
        if not files:
            # Try with _640 suffix (Jetson naming)
            files = sorted(glob.glob('yolo11?_640.onnx'))
        if not files:
            files = sorted(glob.glob('yolo11?.onnx'))

    if not files:
        print("No ONNX files found. Use --models to specify.")
        sys.exit(1)

    measured = {}
    if a.measured:
        measured = json.loads(a.measured)

    results = {}
    for f in files:
        name = os.path.basename(f).replace('.onnx', '').replace('_640', '')
        print(f"  analyzing {f} ...", end=' ')
        try:
            r = analyze_model(f, l3_bytes, measured.get(name))
            results[name] = r
            print(f"{r['n_compute_layers']} layers, "
                  f"{r['layers_over_L3']} > L3 ({r['frac_layers_over_L3']*100:.0f}%)")
        except Exception as e:
            print(f"FAILED: {e}")

    with open(a.output, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\n-> {a.output}")


if __name__ == '__main__':
    main()
