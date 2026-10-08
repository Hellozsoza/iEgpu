# iEgpu

A Linux inference backend that reads a GGUF from the laptop SSD and loads its model weights into a remote worker's RAM.
The native engine is the pinned llama.cpp fork, with RPC weight placement and chunked loading. It serves the usual
OpenAI-compatible API and includes a small command-line client.

**Current scope: Linux backend.** You can build and test it without an iPhone using the included local CPU worker.
The eventual device target is iPhone 15 or newer with iOS 26+. Actual iPhone GPU inference and USB transport are pending
hardware validation; a signed worker app will be required on the phone.

## Build on Fedora

```bash
sudo dnf install git cmake gcc-c++ make python3
git clone --recurse-submodules https://github.com/Hellozsoza/iEgpu.git
cd iEgpu
bash scripts/build-linux.sh
```

The build produces `build/linux-cpu/bin/llama-server` and `ggml-rpc-server`. `./iegpu` runs them through a Python standard
library CLI. No model download, dedicated laptop GPU, Apple tooling, or USB utilities are needed for backend development.
The development target is Fedora 44 on an Intel Core i5-1235U with Iris Xe.

## Run without a phone

Start a separate worker process in one terminal:

```bash
./iegpu worker
```

In a second terminal, load your laptop's model into that worker:

```bash
./iegpu doctor --rpc 127.0.0.1:50052
./iegpu serve --rpc 127.0.0.1:50052 --model ~/Models/model.gguf
```

After the model finishes loading, generate text in a third terminal:

```bash
./iegpu complete --prompt "Explain how a GPU works." --tokens 128
# Or read a prompt from stdin:
printf 'Explain how a GPU works.' | ./iegpu complete
```

The development worker runs on the laptop CPU in a separate process. This exercises real remote inference through RPC;
it does not provide a speedup or additional physical RAM. Replacing that worker with the phone's Metal worker is the
future device integration step.

The API listens at `http://127.0.0.1:8080/v1`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Chat requests need a GGUF with a suitable chat template. Plain completion is available through `./iegpu complete`,
`/completion`, or `/v1/completions`. Use `./iegpu --help` and the subcommands' `--help` for options.

## Weight placement and memory

```text
Laptop SSD -> reads in chunks of up to 8 MiB -> RPC -> worker RAM / compute
                                                   |
Laptop API / tokenizer / sampler <----- results ----+
```

- All model-weight tensors, including embeddings, use the single remote RPC device. Automatic fitting and CPU weight
  splitting are disabled. If the model cannot fit on the worker, loading fails.
- The Linux loader avoids mapping the entire GGUF and stages remote tensor data in chunks of at most 8 MiB. RPC also
  needs transport buffers; this is not an 8 MiB limit on total process RAM. Metadata, tokenization, sampling, API handling,
  and some operations still use the laptop CPU and RAM. The OS may cache source-file reads.
- `iegpu worker` disables the native RPC disk cache. Received weights live in that worker's memory and are transferred
  again for a new session. For a direct `--rpc` endpoint, the worker operator must disable caching; native RPC does not
  report that setting to the host.
- The host disables prompt-state RAM caching and context checkpoints to avoid maintaining extra KV snapshots locally.
  Device KV and compute buffers also consume worker memory.
- A worker failure stops inference. There is no automatic CPU recovery or session resume.

This provides inference through a remote compute backend. It does not register the phone as a general Linux graphics
GPU or give Linux applications direct access to phone RAM.

## Future iPhone connection

The existing USB path is available for integration later:

```bash
# Requires this fork's signed RPC_ONLY=1 phone app and USB pairing:
python3 scripts/serve-linux.py doctor
python3 scripts/serve-linux.py serve --model ~/Models/model.gguf
```

It uses USB-only usbmuxd forwarding, checks that the phone worker disables disk caching, then uses the same Linux engine.
Keep the app open and unlocked. The phone app, signing, USB transport, and Metal inference have not been validated here.
See [the Linux guide](docs/LINUX.md) for the worker contract and the existing phone build path.

## Verification

```bash
python3 -m unittest discover -s tests/linux -v
bash scripts/check-listeners.sh
python3 tests/linux/smoke_rpc.py
```

The offline smoke test creates a random GGUF with tensors larger than 8 MiB and runs the actual `iegpu worker`, `doctor`,
`serve`, and `complete` commands. It compares 16 greedy tokens against a local CPU baseline, checks remote-only weight
buffers and chunked loading, exercises the OpenAI completion API, and checks failure after worker disconnection.
No downloaded model or phone is needed. Temporary model files are removed on exit; logs remain in `build/smoke-*.log`.
The Linux backend workflow runs these checks in CI.

## Credits

iEgpu is a fork of [Backburner](https://github.com/StayLameBro/backburner), created by StayLameBro. It uses the project's
[pinned llama.cpp fork](https://github.com/StayLameBro/backburner-llama.cpp). Both upstream projects use the MIT license;
retain their notices and attribution. Original Mac setup and performance results are available in the upstream README.
