"""从 ONNX 计算图逐层计算工作集,与实测 DRAM 流量对照。"""
import json, os, sys
try:
    import onnx
    from onnx import shape_inference
except ImportError:
    sys.exit("请先: pip install onnx")

L3 = 2*1024*1024          # Pi 5 共享 L3
L2 = 512*1024             # 每核 L2
FP32 = 4

MEASURED = {  # 实测 LLC 缺失/图 × 64 B = MB/图
    'yolo11n': 397, 'yolo11s': 1002, 'yolo11m': 2960,
    'yolo11l': 3601, 'yolo11x': 7967,
}

def numel(shape):
    n = 1
    for d in shape:
        if d is None or d <= 0: return None
        n *= d
    return n

def analyse(path):
    m = shape_inference.infer_shapes(onnx.load(path))
    g = m.graph
    init = {i.name: i for i in g.initializer}

    def dims(t):
        return [d.dim_value if d.HasField('dim_value') else None
                for d in t.type.tensor_type.shape.dim]

    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        shapes[vi.name] = dims(vi)
    for name, t in init.items():
        shapes[name] = list(t.dims)

    weight_bytes = sum(numel(list(t.dims)) * FP32 for t in g.initializer
                       if numel(list(t.dims)))

    rows, unknown = [], 0
    for n in g.node:
        if n.op_type not in ('Conv', 'Gemm', 'MatMul', 'ConvTranspose'):
            continue
        w = a_in = a_out = 0
        for x in n.input:
            s = shapes.get(x)
            if s is None: unknown += 1; continue
            e = numel(s)
            if e is None: unknown += 1; continue
            if x in init: w += e * FP32
            else:         a_in += e * FP32
        for y in n.output:
            s = shapes.get(y)
            if s is None: unknown += 1; continue
            e = numel(s)
            if e is None: unknown += 1; continue
            a_out += e * FP32
        rows.append(dict(op=n.op_type, w=w, a_in=a_in, a_out=a_out, ws=w+a_in+a_out))

    tot = sum(r['ws'] for r in rows)
    over_l3 = [r for r in rows if r['ws'] > L3]
    return dict(
        n_compute_layers=len(rows),
        unknown_shapes=unknown,
        weight_MB=round(weight_bytes/1e6, 2),
        total_workset_MB=round(tot/1e6, 1),
        layers_over_L3=len(over_l3),
        frac_layers_over_L3=round(len(over_l3)/max(len(rows),1), 3),
        MB_in_layers_over_L3=round(sum(r['ws'] for r in over_l3)/1e6, 1),
        frac_MB_over_L3=round(sum(r['ws'] for r in over_l3)/max(tot,1), 3),
        max_layer_ws_MB=round(max((r['ws'] for r in rows), default=0)/1e6, 2),
        median_layer_ws_KB=round(sorted(r['ws'] for r in rows)[len(rows)//2]/1024, 1) if rows else 0,
    )

out = {}
print(f"{'model':<10}{'权重MB':>9}{'工作集和MB':>12}{'层数':>6}{'>L3层数':>9}{'>L3占比':>9}{'实测MB':>9}{'放大':>8}{'预测/实测':>10}")
print("-"*84)
for name in ('yolo11n','yolo11s','yolo11m','yolo11l','yolo11x'):
    p = f"{name}.onnx"
    if not os.path.exists(p):
        print(f"{name:<10}  [缺文件]"); continue
    r = analyse(p)
    meas = MEASURED[name]
    r['measured_MB_per_img'] = meas
    r['dram_amplification'] = round(meas/r['weight_MB'], 1)
    r['pred_over_meas'] = round(r['total_workset_MB']/meas, 3)
    out[name] = r
    print(f"{name:<10}{r['weight_MB']:>9.1f}{r['total_workset_MB']:>12.1f}"
          f"{r['n_compute_layers']:>6}{r['layers_over_L3']:>9}"
          f"{r['frac_layers_over_L3']*100:>8.1f}%{meas:>9}"
          f"{r['dram_amplification']:>7.1f}x{r['pred_over_meas']:>10.3f}")
    if r['unknown_shapes']:
        print(f"{'':<10}  ⚠️ {r['unknown_shapes']} 个张量形状未知(动态轴)")

json.dump(out, open('worksets.json','w'), indent=2)
print("\n-> worksets.json")
