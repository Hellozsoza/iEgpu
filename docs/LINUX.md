# iEgpu on Linux

This fork adds a Linux host launcher and a separate **RAM-only iPhone GPU worker** build.
Target: Fedora 44, Intel Core i5-1235U / Iris Xe, and iPhone 15 or newer running iOS 26 or later.
Device compatibility and speed are **not yet verified on a physical iPhone**. The upstream Mac performance tables do not
describe this mode.

The laptop reads a local GGUF, transfers its weights over USB, and asks the phone's Metal GPU to execute model operations
through llama.cpp RPC. The phone never downloads a GGUF or saves received model tensors to a weight cache. Once the
session ends, the model must be transferred again. This is an inference worker, not a Linux graphics device for other apps.

```text
Laptop SSD -> laptop file reads / transfer buffers -> USB -> iPhone RAM / Metal GPU
                         ^                                  |
                         +---- operations and results ------+
                         |
              local OpenAI-compatible API
```

The source GGUF stays on the laptop SSD. Reading and transferring it still uses laptop RAM and the OS file cache.
The tokenizer, scheduling, sampling, API, and some work buffers also run on the laptop. Default placement overrides
**all model weight tensors, including input embeddings**, to the phone RPC buffer. It does not promise zero laptop RAM
use or that every operation executes on the phone. No `tail.gguf`, Neural Engine page models, or phone filesystem copy
is used. Metal/iOS may maintain their own shader caches; “RAM-only” refers to this program's model weights.

## Hardware and model size

- Use an **iPhone 15 or newer** with a **USB-C-to-USB-C data cable**. iPhone 14 and older are outside this project's target.
- iPhone 15: USB-C, but USB 2, up to 480 Mb/s. A faster cable cannot increase the phone's port speed.
- Faster USB-capable Pro models can reduce transfer time; check the exact phone's specifications. USB-C alone does not
  imply a 10 Gb/s link, and usbmuxd throughput must be measured separately from the port's advertised speed.
- Keep the app in the foreground and the screen unlocked. The current worker has no background GPU entitlement.
- Weights **plus KV cache plus GPU work buffers** must fit the phone's app memory budget. A phone's total physical RAM
  is not its usable app budget. Start with a small quantized GGUF and a 2,048-token context; this does not make the upstream
  27B model fit entirely into a base iPhone. No speedup over your CPU or Iris Xe is guaranteed.

Sources: [iPhone 15 specs](https://support.apple.com/en-us/111831),
[Apple Metal GPU restrictions](https://developer.apple.com/documentation/metal/preparing-your-metal-app-to-run-in-the-background).

## 1. Prepare Fedora

```bash
sudo dnf install git cmake gcc-c++ make python3 usbmuxd libusbmuxd-utils libimobiledevice-utils
git submodule update --init --recursive
bash scripts/build-linux.sh
```

The build uses the repository's pinned llama.cpp revision, RPC, and the host CPU. The build script applies
`patches/llama-linux.patch` for explicit C++ headers and Linux file timestamps. It is safe to rerun, keeps existing local
engine edits, and stops if the patch no longer matches. The submodule will appear modified after applying it.
Downloads of models or UI assets are not required to compile. Build parallelism defaults to at most four jobs;
set `JOBS=2` for a laptop with less free RAM.

Fedora packages: [libusbmuxd-utils](https://packages.fedoraproject.org/pkgs/libusbmuxd/libusbmuxd-utils/),
[libimobiledevice-utils](https://packages.fedoraproject.org/pkgs/libimobiledevice/libimobiledevice-utils/).

## 2. Build and install the phone app once

An installed, signed iOS app is required. A cable and a Linux executable alone cannot execute arbitrary GPU code on iOS.
The upstream Backburner IPA caches RPC weights on storage and is rejected by this launcher.

With a Mac and Xcode 26 or later, build this fork's worker:

```bash
git submodule update --init --recursive
RPC_ONLY=1 IPA=1 bash scripts/build-iphone.sh
# output: ios/build/iEgpu.ipa
```

Without a local Mac, this repo includes a **manual** GitHub Actions workflow, `Build iEgpu iPhone worker`, using an Xcode
runner. After the changes are pushed to your fork, run it from the Actions page and download the `iEgpu-iOS26-worker`
artifact. It builds an unsigned IPA; it does not sign, install, publish, or start an inference session. This workflow has
not been run as part of the Linux host validation.

Sign and sideload the IPA through a method available to you. The existing
[AltStore instructions](INSTALL-IPHONE.md) assume a Mac or Windows computer for installation and renewal. Sideloading
and signing on a Linux-only setup are not implemented by this port. The included free-ID installation route needs
periodic renewal. No Apple ID or signing secrets are needed by the build workflow.

The worker build targets iOS 26+, shows an iEgpu GPU-worker screen, disables disk caching at the native RPC entry point,
and starts only the GPU RPC service and read-only memory-report control service. It does not start the Mac split-prefill
tail, SME attention, ANE model workers, or Wi-Fi tunnel, so those newer-chip kernels are not used on an A16 phone.
The general Metal backend chooses device-supported kernels. This is the intended compatibility path, not a physical-device
certification for every iPhone 15+ or every future iOS version.

## 3. Plug in and check the connection

Unlock the phone, connect its data cable, tap **Trust**, and open iEgpu. Pair with the host if needed:

```bash
idevice_id -l
idevicepair pair
python3 scripts/serve-linux.py doctor
```

`doctor` starts temporary, USB-only `iproxy` forwards for ports 50052 and 50061, verifies the RAM-only worker flags, and
uses the host engine to perform a GPU RPC handshake. It stops its forwarding process on exit. It neither copies models
nor runs inference. With several USB iOS devices, use `--udid` to select one of the IDs printed by `idevice_id -l`.

The transport uses **usbmuxd**, not Personal Hotspot or Wi-Fi. No network-interface configuration, link-local address,
broadcast ping, `ioreg`, or `xcrun` is needed on Linux. The app's existing cable/loopback connection gates stay in place.
How usbmuxd presents the connection to that gate must still be confirmed on the target iOS/device combinations;
if the app refuses it, collect diagnostics instead of disabling the gate.

Source: [libusbmuxd iproxy](https://github.com/libimobiledevice/libusbmuxd/blob/master/tools/iproxy.c).

## 4. Load your laptop model and serve inference

```bash
python3 scripts/serve-linux.py serve --model ~/Models/model.gguf
```

The launcher binds both forwarding ports and the API to loopback. Default settings are all weight tensors on the phone,
2,048 context tokens, and one request at a time. Automatic memory fitting is disabled so an oversized model does not
silently become a CPU-weight split. Check the engine's loading log: model weight buffers should use `RPC0[...]`.
The API becomes available at `http://127.0.0.1:8080/v1` only after loading finishes.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Change context or API port with `--ctx` and `--port`. For an **explicit CPU/phone weight split**, use a finite
`--gpu-layers` value such as `--gpu-layers 8`: that removes the all-tensor override and keeps remaining layers and input
embeddings on the laptop. It still never enables the phone's disk weight cache.
Ctrl-C stops the host and its USB forwarding process. Reconnect and restart after unplugging, locking, or an iOS app
termination; there is no automatic resume or CPU recovery of a phone-held session.

An optional Iris Xe/Vulkan host build is available with `BACKEND=vulkan bash scripts/build-linux.sh`, using a Vulkan SDK
and shader compiler. Pass `--server build/linux-vulkan/bin/llama-server` to the launcher. Its selected inference GPU remains
the phone; combined Iris Xe/phone GPU placement is not implemented or benchmarked by the launcher.

RPC background and cache behavior: [llama.cpp RPC documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md).
Use the same pinned engine for the app and host: an arbitrary system llama-server can have an incompatible RPC protocol.

## Validation and next device checks

Host validation on Fedora 44 / GCC 16.2.1: `llama-server` and `ggml-rpc-server` build successfully, all eight launcher tests
pass, static listener checks pass, and the offline loopback smoke test matches 16/16 greedy tokens with every weight tensor
in the remote RPC buffer. The iOS app build, signing, physical USB transport, and iPhone inference have not been tested here.

Run host tests with `python3 -m unittest discover -s tests/linux -v` and static network checks with
`bash scripts/check-listeners.sh`. The original `tests/security/run.sh` also needs Xcode's Swift/C++ tools and should run
on a Mac before distributing a phone build. Physical phone exposure tests remain required for distribution.

For an offline end-to-end host check, build `ggml-rpc-server` in the same Linux build directory and run the fixture:

```bash
cmake --build build/linux-cpu --target ggml-rpc-server -j 4
python3 tests/linux/smoke_rpc.py
```

It creates a tiny random GGUF, loads it through local RPC with caching disabled, generates 16 greedy tokens through the
API, compares them with a local CPU run, and checks remote weight placement. The generated model is deleted on exit;
logs stay in `build/smoke-*.log`. This tests the remote CPU RPC path, not an iPhone or the USB cable.

Before calling this a supported device, build/sign/install the worker, pass `doctor`, generate with a small GGUF,
check that no model cache files appear in the phone's app container, compare greedy output against a CPU run, and measure
load time and generation speed. Also test app locking, cable disconnection, and model-too-large failures. Record the exact
phone, iOS version, model, context, and build revision. Host-only or loopback RPC tests do not establish iPhone compatibility.
