# iEgpu

Use an iPhone as a USB-connected inference worker for a Linux laptop. Model weights load from the laptop's SSD into the
phone's RAM, and the phone's Metal GPU runs model operations through llama.cpp RPC. The phone does not store a model copy
or a model-weight disk cache.

**Experimental.** The Linux engine builds and passes host tests on Fedora 44. The iOS worker build and inference over USB
have not yet been verified on a physical iPhone. iPhone 15 or newer with iOS 26 or later is the compatibility target.

## Requirements

| Component | Requirement |
| --- | --- |
| Laptop | Linux; Fedora 44 is the current development target |
| Host hardware | A C++17-capable build environment; target laptop: Intel Core i5-1235U with Iris Xe |
| Phone | iPhone 15 or newer, running iOS 26 or later; base and Pro models are targets |
| Cable | USB-C-to-USB-C **data** cable |
| Phone app | This fork's signed iEgpu worker, open in the foreground with the phone unlocked |
| Model | A local GGUF whose weights, KV cache, and work buffers fit the phone's app memory budget |

A dedicated laptop GPU is not required. The default host build uses the laptop CPU for orchestration and the phone GPU
for offloaded computation. iPhone 14 and older are outside this project's target.

USB-C does not guarantee a fast link: the base [iPhone 15 supports USB 2, up to 480 Mb/s](https://support.apple.com/en-us/111831).
A faster cable cannot increase a phone's port speed. Actual transfer speed through usbmuxd still needs measurement.

## How it works

```text
Laptop SSD -> laptop transfer buffers -> USB -> iPhone RAM / Metal GPU
                    ^                                  |
                    +-------- operations/results -------+
                    |
         Local OpenAI-compatible API
```

The source GGUF stays on the laptop. Reading it still uses laptop RAM and the OS file cache. Tokenization, scheduling,
sampling, and the API run on the laptop too. Default placement puts all model-weight tensors, including input embeddings,
in the phone's RPC buffer; it does not guarantee that every operation runs on the phone.

The worker disables the native RPC weight cache and does not load a `tail.gguf` or Neural Engine model files. Weights
must be transferred again for a new session. iOS and Metal may maintain their own shader caches; RAM-only refers to the
program's model weights.

## Set up the Linux host

On Fedora 44:

```bash
sudo dnf install git cmake gcc-c++ make python3 usbmuxd libusbmuxd-utils libimobiledevice-utils
git clone --recurse-submodules https://github.com/Hellozsoza/iEgpu.git
cd iEgpu
bash scripts/build-linux.sh
```

The build uses the pinned llama.cpp dependency and applies the included Linux portability patch. It does not download
models. See [the Linux guide](docs/LINUX.md) for build settings and optional Vulkan support.

## Install the iPhone worker

An installed, signed iOS app is required. The upstream Backburner IPA enables model-weight caching on phone storage, so
the Linux launcher rejects it for this mode.

To build without a local Mac, open this fork's **Actions** page, select **Build iEgpu iPhone worker**, and run the manual
workflow. Download the `iEgpu-iOS26-worker` artifact when it finishes. It contains an unsigned IPA that must be signed
and sideloaded before use. This workflow has not yet been validated by a completed run.

With a Mac and Xcode 26 or later, build the same worker locally:

```bash
RPC_ONLY=1 IPA=1 bash scripts/build-iphone.sh
# Output: ios/build/iEgpu.ipa
```

[Installation details](docs/LINUX.md#2-build-and-install-the-phone-app-once) cover the current signing and sideloading limits.
The existing AltStore instructions use a Mac or Windows computer; Linux-only signing and installation are not implemented
by this port. The worker build requires no Apple ID or signing secrets to produce its unsigned IPA.

## Connect and run

Connect the USB-C data cable, unlock the phone, tap **Trust**, and open iEgpu. If pairing is needed:

```bash
idevice_id -l
idevicepair pair
```

Check the phone's RAM-only worker settings and GPU RPC service:

```bash
python3 scripts/serve-linux.py doctor
```

Load a GGUF from your laptop:

```bash
python3 scripts/serve-linux.py serve --model ~/Models/model.gguf
```

After loading, the OpenAI-compatible API is available at `http://127.0.0.1:8080/v1`. The launcher uses USB-only usbmuxd
forwarding and binds its local ports to loopback. It does not need Wi-Fi or Personal Hotspot.

Defaults: all weight tensors on the phone, 2,048 context tokens, and one request at a time. Automatic memory fitting is
disabled, so an oversized model does not silently become a CPU-weight split. Use `--ctx` or `--port` to change those
settings. A finite `--gpu-layers` value, such as `--gpu-layers 8`, explicitly keeps the remaining layers and embeddings
on the laptop. Use `--udid` when several USB iOS devices are connected.

## Current limits

- Phone weights, KV cache, and work buffers must fit its **app memory budget**, which is smaller than its physical RAM.
  Start with a small quantized model. The upstream 27B model cannot be assumed to fit entirely on a base iPhone.
- Keep the app open and the phone unlocked. Locking, app termination, or cable disconnection can stop inference; reconnect
  and restart the host. There is no automatic session recovery.
- Compatibility across iPhone 15+ devices and iOS 26+ versions remains unverified. No speedup over the laptop CPU or
  Iris Xe has been demonstrated.
- This exposes model-compute services, not a general Linux graphics device. The Mac-specific split-prefill and Neural
  Engine acceleration modes are not used by the Linux worker.

## Validation

Completed on Fedora 44 with GCC 16.2.1:

- Linux `llama-server` and `ggml-rpc-server` builds.
- Eight launcher tests covering USB selection, RAM-only checks, protocol replies, and process cleanup.
- Static network-listener checks.
- An offline inference check using a tiny generated GGUF: all weights in remote RPC memory and **16/16 greedy tokens
  matching a local CPU run**.

The offline check uses a remote CPU over loopback, so it does not establish USB transport or iPhone Metal compatibility.
The iOS build, signing, physical-phone exposure checks, and real-phone inference are still pending.

```bash
python3 -m unittest discover -s tests/linux -v
bash scripts/check-listeners.sh
cmake --build build/linux-cpu --target ggml-rpc-server -j 4
python3 tests/linux/smoke_rpc.py
```

See [docs/LINUX.md](docs/LINUX.md) for the full setup, troubleshooting notes, and remaining device checks.

## Credits

iEgpu is a fork of [Backburner](https://github.com/StayLameBro/backburner), created by StayLameBro. It uses the project's
[pinned llama.cpp fork](https://github.com/StayLameBro/backburner-llama.cpp). Both upstream projects use the MIT license;
retain their notices and attribution. Original Mac setup and performance results are available in the upstream README.
