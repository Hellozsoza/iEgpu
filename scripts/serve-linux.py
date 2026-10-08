#!/usr/bin/env python3
"""Linux -> USB-only usbmuxd tunnel -> iPhone Metal RPC, without a phone weight cache.

    python3 scripts/serve-linux.py doctor
    python3 scripts/serve-linux.py serve --model ~/Models/model.gguf

Requires the RPC-only iOS app built from this tree, libimobiledevice, and iproxy.
Uses Python's standard library. Does not install, pair, or launch apps on the phone.
"""

import argparse
import contextlib
import json
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
                            text=True, capture_output=True, timeout=30, check=True)
    names = re.findall(r"^\s*(RPC[^\s:]*):", result.stdout + "\n" + result.stderr, re.MULTILINE)
    if len(names) != 1:
        raise WorkerError("Expected one iPhone RPC GPU. Check the phone's GPU service and ensure "
                          "the host and app use the same pinned llama.cpp revision.")
    return names[0]


def server_command(args, endpoint, device):
    command = [str(args.server), "--model", str(args.model), "--rpc", endpoint,
            "--device", device, "--n-gpu-layers", str(args.gpu_layers),
            "--fit", "off", "--ctx-size", str(args.ctx), "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(args.port), "--log-verbosity", "4"]
    if args.gpu_layers == 999:
        # This pinned engine otherwise keeps input embeddings on the laptop CPU.
        # One remote server, one exposed GPU: its buffer type is RPC0[endpoint].
        command.extend(["--override-tensor", f".=RPC0[{endpoint}]"])
    return command


def run_server(command, tunnel):
    env = os.environ.copy()
    env["LLAMA_LAZY_EMBD"] = "0"  # a Mac tuning setting would keep embeddings mapped on the host
    process = subprocess.Popen(command, env=env)
    try:
        while process.poll() is None:
            if tunnel.poll() is not None:
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


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("doctor", "serve"):
        p = sub.add_parser(action)
        p.add_argument("--udid", help="select one USB device (idevice_id -l)")
        p.add_argument("--server", type=Path, default=ROOT / "build/linux-cpu/bin/llama-server")
        if action == "serve":
            p.add_argument("--model", type=Path, required=True, help="local GGUF on the laptop SSD")
            p.add_argument("--ctx", type=positive, default=2048)
            p.add_argument("--gpu-layers", type=positive, default=999,
                           help="layers on the phone; reduce for a CPU/phone split")
            p.add_argument("--port", type=port_number, default=8080)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise WorkerError("Run this launcher on Linux.")
    args.server = args.server.expanduser().resolve()
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise WorkerError("Build the host engine first: bash scripts/build-linux.sh")
    if args.action == "serve":
        args.model = args.model.expanduser().resolve()
        if not args.model.is_file():
            raise WorkerError("Model file does not exist; pass a local GGUF with --model.")
        with args.model.open("rb") as model:
            if model.read(4) != b"GGUF":
                raise WorkerError("The model must be a GGUF file.")
    udid = choose_phone(args.udid)
    with usb_tunnel(udid) as (endpoint, info, tunnel):
        device = remote_device(args.server, endpoint)
        print(f"USB iPhone GPU: {device}; phone weight cache: off", flush=True)
        if "avail_mb" in info:
            print(f"Phone app memory available: {info['avail_mb']} MiB (also needed by KV and work buffers)", flush=True)
        if args.action == "doctor":
            print("USB control and GPU RPC handshakes passed. Inference still needs a model/hardware test.")
            return 0
        print(f"Loading laptop GGUF over USB; API at http://127.0.0.1:{args.port}/v1 after loading.", flush=True)
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
        print(f"serve-linux: {error}", file=sys.stderr)
        sys.exit(1)
