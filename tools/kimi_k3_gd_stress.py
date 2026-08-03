#!/usr/bin/env python3
"""Изолированный стресс gated-delta Metal-кернела (#1626) под K3-декод-профиль.

Цель: детерминированно воспроизвести GPU Address Fault вне EXO — или
оправдать кернел в изоляции (тогда виновник = связка с async-конвейером
opt_batch_gen, проверяется --inflight).

Запуск НА НОДЕ (Metal), single-node, меш не нужен:
  .venv/bin/python tools/kimi_k3_gd_stress.py --iters 20000            # базовый T=1
  .venv/bin/python tools/kimi_k3_gd_stress.py --iters 20000 --inflight 8   # с конвейером
  .venv/bin/python tools/kimi_k3_gd_stress.py --iters 300 --t 512      # prefill-профиль
Падение процесса (Abort trap: 6 / address fault в консоли) = репро пойман.
Выживание всех режимов при --verify без расхождений = кернел чист в изоляции.
"""
from __future__ import annotations
import argparse, sys, time
import mlx.core as mx

sys.path.insert(0, "src")
from exo.worker.engines.mlx.vendor.kimi_k3_gated_delta import (  # noqa: E402
    gated_delta_update, gated_delta_ops, compute_g_safe,
)

def run(a):
    B, H, Dk, Dv = 1, a.heads, 128, 128
    lb = -5.0
    mx.random.seed(7)
    A_log = mx.random.normal((H, 1)) * 0.5
    dt_bias = (mx.random.normal((H, Dk)) * 0.5).astype(mx.float32)
    state_k = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)
    state_o = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)
    inflight: list[mx.array] = []
    t0 = time.time()
    for it in range(a.iters):
        T = a.t
        q = (mx.random.normal((B, T, H, Dk)) * 0.05).astype(mx.bfloat16)
        k = (mx.random.normal((B, T, H, Dk)) * 0.05).astype(mx.bfloat16)
        v = (mx.random.normal((B, T, H, Dv)) * 0.05).astype(mx.bfloat16)
        aa = (mx.random.normal((B, T, H, Dk)) * 2.0).astype(mx.bfloat16)
        bb = (mx.random.normal((B, T, H)) * 2.0).astype(mx.bfloat16)
        y, state_k = gated_delta_update(
            q, k, v, aa, bb, A_log, dt_bias,
            state=state_k, mask=None, use_kernel=True, lower_bound=lb,
        )
        if a.inflight:
            # имитация opt_batch_gen: держим до N eval'ов в полёте
            inflight.append(y)
            if len(inflight) >= a.inflight:
                mx.eval(inflight.pop(0))
        else:
            mx.eval(y, state_k)
        if a.verify and (it % a.verify == 0):
            g = compute_g_safe(A_log, aa.astype(mx.float32)
                               + 0 * dt_bias.reshape(1, 1, H, Dk), dt_bias * 0, lb)
            # честная сверка: тот же g, что видел кернел
            g = compute_g_safe(A_log, aa.astype(mx.float32), dt_bias, lb)
            y2, state_o = gated_delta_ops(q, k, v, g, mx.sigmoid(bb.astype(mx.float32)),
                                          state_o, None)
            mx.eval(y2, state_o)
            d = float(mx.abs(y.astype(mx.float32) - y2.astype(mx.float32)).max())
            sd = float(mx.abs(state_k - state_o).max())
            if d > 3e-2 or sd > 3e-2:
                print(f"[stress] MISMATCH it={it}: |dy|={d:.4f} |dstate|={sd:.4f}")
                return 3
        if it and it % 1000 == 0:
            el = time.time() - t0
            print(f"[stress] it={it} ok, {it/el:.0f} it/s", flush=True)
    for x in inflight:
        mx.eval(x)
    print(f"[stress] DONE {a.iters} iters, T={a.t}, inflight={a.inflight}: кернел выжил")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--heads", type=int, default=24)  # = 96/4, наш шард
    ap.add_argument("--inflight", type=int, default=0)
    ap.add_argument("--verify", type=int, default=500,
                    help="каждые N итераций сверять с ops (0 = выкл)")
    sys.exit(run(ap.parse_args()))
