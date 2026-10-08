#!/usr/bin/env python3
"""Offline SSD -> RPC memory -> inference/API check (remote CPU, not an iPhone).

Build llama-server and ggml-rpc-server first, then:
    python3 tests/linux/smoke_rpc.py

Creates a tiny random GGUF with Python's standard library, checks greedy tokens
against a local CPU run, and removes the model on exit. No model downloads.
"""
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import random
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build/linux-cpu/bin"
spec = importlib.util.spec_from_file_location("worker", ROOT / "scripts/serve-linux.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def tiny_model(path):
    # GGUF v3: one Llama layer, 32 embedding dimensions, byte-level GPT-2 vocab.
    byte_chars = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in byte_chars}
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + len(mapping) - len(byte_chars))
    tokens = [mapping[b] for b in range(256)] + ["<s>", "</s>"]
    meta = {
        "general.architecture": (8, "llama"), "general.name": (8, "iEgpu RPC smoke fixture"),
        "llama.context_length": (4, 128), "llama.embedding_length": (4, 32),
        "llama.block_count": (4, 1), "llama.feed_forward_length": (4, 64),
        "llama.attention.head_count": (4, 4), "llama.attention.head_count_kv": (4, 4),
        "llama.attention.layer_norm_rms_epsilon": (6, 1e-5),
        "llama.rope.dimension_count": (4, 8),
        "tokenizer.ggml.model": (8, "gpt2"), "tokenizer.ggml.pre": (8, "gpt-2"),
        "tokenizer.ggml.tokens": (9, (8, tokens)), "tokenizer.ggml.merges": (9, (8, [])),
        "tokenizer.ggml.token_type": (9, (5, [1] * 256 + [3, 3])),
        "tokenizer.ggml.bos_token_id": (4, 256), "tokenizer.ggml.eos_token_id": (4, 257),
        "tokenizer.ggml.add_bos_token": (7, False),
    }
    shapes = {"token_embd.weight": (32, 258), "output_norm.weight": (32,),
              "output.weight": (32, 258), "blk.0.attn_norm.weight": (32,),
              "blk.0.ffn_norm.weight": (32,), "blk.0.ffn_gate.weight": (32, 64),
              "blk.0.ffn_up.weight": (32, 64), "blk.0.ffn_down.weight": (64, 32)}
    for name in ("q", "k", "v", "output"):
        shapes[f"blk.0.attn_{name}.weight"] = (32, 32)

    def value(kind, data):
        if kind == 8:
            return string(data)
        if kind == 9:
            item_type, items = data
            return struct.pack("<IQ", item_type, len(items)) + b"".join(value(item_type, i) for i in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], data)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), len(meta))
    for key, (kind, data) in meta.items():
        header += string(key) + struct.pack("<I", kind) + value(kind, data)
    rng = random.Random(17)
    payload = bytearray()
    for name, dims in shapes.items():
        payload.extend(b"\0" * (-len(payload) % 32))
        header += string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", 0, len(payload))
        values = [1.0 if "norm" in name else rng.uniform(-0.2, 0.2) for _ in range(math.prod(dims))]
        payload.extend(struct.pack("<" + "f" * len(values), *values))
    path.write_bytes(header + b"\0" * (-len(header) % 32) + payload)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def process(command, log):
    child = subprocess.Popen(command, stdout=log, stderr=log)
    try:
        yield child
    finally:
        worker.stop_process(child)


def tokens(port, child):
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if child.poll() is not None:
            raise RuntimeError("Inference server stopped; inspect its smoke log.")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    else:
        raise RuntimeError("Server health timeout")
    data = json.dumps({"prompt": "Hello", "n_predict": 16, "temperature": 0,
                       "ignore_eos": True, "return_tokens": True}).encode()
    request = urllib.request.Request(base + "/completion", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert result["tokens_predicted"] == 16, result
    assert len(result["tokens"]) == 16, result
    return result["tokens"]


def main():
    logs = ROOT / "build"
    with tempfile.TemporaryDirectory(prefix="iegpu-smoke-") as tmp, contextlib.ExitStack() as stack:
        model = Path(tmp) / "fixture.gguf"
        tiny_model(model)
        rpc_log = stack.enter_context((logs / "smoke-rpc.log").open("w"))
        cpu_log = stack.enter_context((logs / "smoke-cpu.log").open("w"))
        host_log = stack.enter_context((logs / "smoke-host.log").open("w"))
        endpoint = f"127.0.0.1:{free_port()}"
        rpc = stack.enter_context(process([str(BIN / "ggml-rpc-server"), "--host", "127.0.0.1",
                                          "--port", endpoint.rsplit(":", 1)[1]], rpc_log))
        for _ in range(100):
            if rpc.poll() is not None:
                raise RuntimeError("RPC server stopped; inspect smoke-rpc.log")
            try:
                with socket.create_connection(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        device = worker.remote_device(BIN / "llama-server", endpoint)
        common = [str(BIN / "llama-server"), "-m", str(model), "-c", "128", "-np", "1",
                  "--fit", "off", "--host", "127.0.0.1", "--no-warmup", "--log-verbosity", "4"]
        cpu_port, host_port = free_port(), free_port()
        cpu = stack.enter_context(process(common + ["--device", "none", "-ngl", "0", "--port", str(cpu_port)], cpu_log))
        expected = tokens(cpu_port, cpu)
        command = common + ["--rpc", endpoint, "--device", device, "-ngl", "999",
                            "--override-tensor", f".=RPC0[{endpoint}]", "--port", str(host_port)]
        host = stack.enter_context(process(command, host_log))
        actual = tokens(host_port, host)
        assert expected == actual, (expected, actual)
        host_log.flush()
        log_text = (logs / "smoke-host.log").read_text()
        assert f"RPC0[{endpoint}] model buffer" in log_text, "No remote weight buffer found in the loading log"
        assert "CPU_Mapped model buffer" not in log_text, "Unexpected host weight buffer"
        print(f"Loopback RPC: 16/16 greedy tokens match local CPU. Model bytes: {model.stat().st_size}.")
        print("This verifies the Linux engine path, not USB transport or iPhone Metal compatibility.")


if __name__ == "__main__":
    main()
