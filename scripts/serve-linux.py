#!/usr/bin/env python3
"""iEgpu: a Linux GGUF inference engine with weights on a remote RPC worker.

    ./iegpu worker
    ./iegpu serve --rpc 127.0.0.1:50052 --model ~/Models/model.gguf
    ./iegpu complete --prompt "Hello"

The local CPU worker tests the protocol without a phone. Omit --rpc for the
future USB iPhone worker (requires its app, libimobiledevice, and iproxy).
The native engine is the pinned llama.cpp; this CLI uses Python's standard library.
"""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


class WorkerError(Exception):
    pass


def require_tool(name):
    path = shutil.which(name)
    if not path:
        raise WorkerError(f"Missing {name}. See docs/LINUX.md for Fedora dependencies.")
    return path


def choose_phone(udid=None):
    # idevice_id -l lists USB devices; -n (network discovery) is deliberately absent.
    result = subprocess.run([require_tool("idevice_id"), "-l"],
                            text=True, capture_output=True, timeout=10, check=True)
    devices = result.stdout.split()
    if udid:
        if udid not in devices:
            raise WorkerError("The selected phone is not connected over USB.")
        return udid
    if not devices:
        raise WorkerError("No USB iPhone found. Unlock it, tap Trust, and check usbmuxd.")
    if len(devices) != 1:
        raise WorkerError("Multiple USB devices found. Select one with --udid (idevice_id -l).")
    return devices[0]


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def control_memory(port):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
        conn.sendall(b"mem\n")
        with conn.makefile("rb") as stream:
            line = stream.readline(65537)
    if not line.endswith(b"\n") or len(line) > 65536:
        raise WorkerError("Invalid reply from the phone control service.")
    try:
        data = json.loads(line)
    except (ValueError, UnicodeError) as error:
        raise WorkerError("Invalid JSON from the phone control service.") from error
    if not isinstance(data, dict) or data.get("error"):
        raise WorkerError("The phone refused its memory report.")
    return data


def require_ram_worker(info):
    # An older release silently caches RPC weights on phone storage. Fail closed.
    if info.get("rpc_cache_enabled") is not False:
        raise WorkerError("Phone weight caching is enabled or unknown. Install this fork's "
                          "RPC_ONLY=1 app; an upstream IPA does not guarantee RAM-only weights.")
    if info.get("rpc_only") is not True:
        raise WorkerError("Install the RPC_ONLY=1 app for the Linux worker. It disables the "
                          "Mac-specific tail, SME attention, and ANE model loaders.")


@contextlib.contextmanager
def usb_tunnel(udid, timeout=20):
    iproxy = require_tool("iproxy")
    # Reserve ports while choosing them. iproxy must bind them itself, so close
    # immediately before launch; readiness checks below detect a failed bind.
    with contextlib.ExitStack() as stack:
        ports = []
        for _ in range(2):
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    rpc_port, control_port = ports
    with tempfile.TemporaryFile(mode="w+b") as log:
        # Explicit USB-only selection and loopback bind: no Wi-Fi discovery or LAN listener.
        process = subprocess.Popen([iproxy, "-l", "-u", udid, "-s", "127.0.0.1",
                                    f"{rpc_port}:50052", f"{control_port}:50061"],
                                   stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + timeout
            last_error = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise WorkerError("iproxy stopped. Use libusbmuxd 2.x with -l/-s and port-pair support.")
                try:
                    info = control_memory(control_port)
                    break
                except (OSError, WorkerError) as error:
                    last_error = error
                    time.sleep(0.2)
            else:
                raise WorkerError("Phone control service did not answer over USB. Keep the app "
                                  f"open and unlocked; check Trust/pairing. ({last_error})")
            require_ram_worker(info)
            yield f"127.0.0.1:{rpc_port}", info, process
        finally:
            stop_process(process)


def remote_device(server, endpoint):
    # Use the real engine handshake instead of guessing the RPC version or device name.
    result = subprocess.run([str(server), "--rpc", endpoint, "--list-devices"],
                            text=True, capture_output=True, timeout=30, check=True,
                            env=engine_environment())
    names = re.findall(r"^\s*(RPC[^\s:]*):", result.stdout + "\n" + result.stderr, re.MULTILINE)
    if len(names) != 1:
        raise WorkerError("Expected one RPC device. Expose one worker device and use the same "
                          "pinned llama.cpp revision on both ends.")
    return names[0]


def server_command(args, endpoint, device):
    command = [str(args.server), "--model", str(args.model), "--rpc", endpoint,
            "--device", device, "--n-gpu-layers", "999",
            "--fit", "off", "--ctx-size", str(args.ctx), "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(args.port), "--log-verbosity", "4",
            "--load-mode", "none", "--lazy-mode", "off",
            "--cache-ram", "0", "--ctx-checkpoints", "0",
            "--override-tensor", f".=RPC0[{endpoint}]"]
    return command


def engine_environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
    env["LLAMA_LAZY_EMBD"] = "0"  # a Mac tuning setting would keep embeddings mapped on the host
    return env


def run_server(command, tunnel=None):
    process = subprocess.Popen(command, env=engine_environment())
    try:
        while process.poll() is None:
            if tunnel is not None and tunnel.poll() is not None:
                raise WorkerError("The USB tunnel stopped; stopping inference. Reconnect and restart.")
            time.sleep(0.2)
        return process.returncode
    finally:
        stop_process(process)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def port_number(value):
    number = positive(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return number


def rpc_endpoint(value):
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("use 127.0.0.1:PORT (a local worker or USB/SSH forward)")
    return f"127.0.0.1:{port_number(match[1])}"


def temperature(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def complete(args):
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    data = json.dumps({"prompt": prompt, "n_predict": args.tokens,
                       "temperature": args.temperature, "cache_prompt": True}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}/completion", data=data,
                                     headers={"Content-Type": "application/json"})
    # Ignore proxy environment variables for the local API.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise WorkerError(f"Inference API returned HTTP {error.code}; check the server log.") from error
    except urllib.error.URLError as error:
        raise WorkerError("Local API unavailable. Start 'iegpu serve' and wait for /health first.") from error
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise WorkerError("Invalid completion response from the inference API.")
    print(json.dumps(result) if args.json else result["content"])
    return 0


def parser():
    cli = argparse.ArgumentParser(prog="iegpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("doctor", "serve"):
        p = sub.add_parser(action)
        transport = p.add_mutually_exclusive_group()
        transport.add_argument("--udid", help="select one USB device (idevice_id -l)")
        transport.add_argument("--rpc", type=rpc_endpoint,
                               help="direct loopback RPC endpoint; worker must have disk caching disabled")
        p.add_argument("--server", type=Path, default=ROOT / "build/linux-cpu/bin/llama-server")
        if action == "serve":
            p.add_argument("--model", type=Path, required=True, help="local GGUF on the laptop SSD")
            p.add_argument("--ctx", type=positive, default=2048)
            p.add_argument("--port", type=port_number, default=8080)
    p = sub.add_parser("worker", help="run a cache-free local CPU worker for backend development")
    p.add_argument("--port", type=port_number, default=50052)
    p.add_argument("--threads", type=positive, default=min(4, os.cpu_count() or 1))
    p.add_argument("--worker", type=Path, default=ROOT / "build/linux-cpu/bin/ggml-rpc-server")
    p = sub.add_parser("complete", help="generate text through a running local iEgpu server")
    p.add_argument("--prompt", "-p", help="prompt text; read stdin when omitted")
    p.add_argument("--tokens", "-n", type=positive, default=128)
    p.add_argument("--temperature", type=temperature, default=0.8)
    p.add_argument("--port", type=port_number, default=8080)
    p.add_argument("--json", action="store_true", help="print the full completion result")
    return cli


def executable(path):
    path = path.expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise WorkerError("Build the host engine first: bash scripts/build-linux.sh")
    return path


@contextlib.contextmanager
def connection(args):
    if args.rpc:
        # Native RPC does not report its cache policy. USB mode verifies it through
        # the phone control service; a direct worker is configured by its operator.
        yield args.rpc, {}, None
    else:
        with usb_tunnel(choose_phone(args.udid)) as transport:
            yield transport


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise WorkerError("Run this launcher on Linux.")
    if args.action == "complete":
        return complete(args)
    if args.action == "worker":
        binary = executable(args.worker)
        print(f"Development CPU worker: 127.0.0.1:{args.port}; model disk cache: off", flush=True)
        return run_server([str(binary), "--host", "127.0.0.1", "--port", str(args.port),
                           "--device", "CPU", "--threads", str(args.threads)])
    args.server = executable(args.server)
    if args.action == "serve":
        args.model = args.model.expanduser().resolve()
        if not args.model.is_file():
            raise WorkerError("Model file does not exist; pass a local GGUF with --model.")
        with args.model.open("rb") as model:
            if model.read(4) != b"GGUF":
                raise WorkerError("The model must be a GGUF file.")
    with connection(args) as (endpoint, info, tunnel):
        device = remote_device(args.server, endpoint)
        print(f"Remote device: {device} at {endpoint}; all model weights placed remotely", flush=True)
        print("Worker disk cache: verified off" if info else
              "Direct RPC: cache policy is configured by the worker operator (iegpu worker disables it).", flush=True)
        if "avail_mb" in info:
            print(f"Phone app memory available: {info['avail_mb']} MiB (also needed by KV and work buffers)", flush=True)
        if args.action == "doctor":
            print("RPC handshake passed. No model loaded.")
            return 0
        print(f"Streaming laptop GGUF to RPC memory; API at http://127.0.0.1:{args.port}/v1 after loading.", flush=True)
        return run_server(server_command(args, endpoint, device), tunnel)


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (WorkerError, OSError, subprocess.SubprocessError) as error:
        print(f"iegpu: {error}", file=sys.stderr)
        sys.exit(1)
