#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SidecarRPC : NSObject
/// ggml RPC server: listens on 127.0.0.1:(port + 1000) only; a cable-only gate owns `port` (`host` is ignored).
/// Blocking. Call off the main thread. Returns nil on clean exit (it does not return while listening).
/// An empty cacheDir disables disk caching. RPC_ONLY builds always disable it.
+ (nullable NSString *)startHost:(NSString *)host port:(int)port cacheDir:(NSString *)cacheDir
    NS_SWIFT_NAME(start(host:port:cacheDir:));
/// The USB-cable address only: one 169.254 IPv4, or empty. Wi-Fi and IPv6 are not returned.
+ (NSString *)cableAddress;

/// The Wi-Fi pairing key (32 bytes, this device's Keychain), or nil when not paired. See Tunnel.swift.
+ (nullable NSData *)wifiKey;
/// What the Wi-Fi tunnel is doing, for the control port's `mem` report.
+ (void)setWifiTunnelStatus:(NSString *)status;

/// The Wi-Fi IPv4 address, or empty (diagnostics: the exposure check connects to it from the Mac).
+ (NSString *)wifiAddress;

/// Metal device stats: deviceName, allocatedBytes, recommendedWorkingSetBytes, hasUnifiedMemory.
+ (NSDictionary<NSString *, id> *)metalStats;

/// Process/system memory: availableBytes (os_proc_available_memory), footprintBytes, physicalBytes.
+ (NSDictionary<NSString *, id> *)memoryStats;

/// Cumulative rx/tx bytes and packets across non-loopback interfaces. Caller differentiates
/// to get a rate - this is how you tell from the phone whether the Mac is actually using it.
+ (NSDictionary<NSString *, id> *)linkStats;

/// Split-prefill tail worker (llama.cpp tools/split-prefill/tail-server.h). Blocking; call off the
/// main thread. Loads the TAIL GGUF at modelPath (if present) and serves the Mac on port.
+ (nullable NSString *)startTailPort:(int)port modelPath:(NSString *)modelPath
    NS_SWIFT_NAME(startTail(port:modelPath:));

/// Tail worker status: state, detail, model, chunks, tokens, sessions, lastChunkMs, lastTokS.
+ (NSDictionary<NSString *, id> *)tailStatus;

/// Hardware model string (utsname.machine).
+ (NSString *)deviceModel;
@end

@interface SidecarRPC (ANE)
/// ANE bench server (line protocol, JSON replies) on port: loads Documents/*.mlmodelc via CoreML and times it.
+ (void)startANEBenchPort:(int)port NS_SWIFT_NAME(startANEBench(port:));
/// Phone-held KV attention service (phone-attn/phone-attn.h) on port (50062), computed with SME2. Starts once.
+ (void)startPhoneAttnPort:(int)port NS_SWIFT_NAME(startPhoneAttn(port:));
/// Phone attention status: state, calls, heldKeys, lastMs.
+ (NSDictionary<NSString *, id> *)phoneAttnStatus;
// what the Mac is doing, as its proxy and serve-infernet.sh report it on :50061 ("mac PHASE [N1] [N2] [CTX]"):
// phase, n1, n2, ctx, age (s since the last report), activations (count of "ready"), activating, activateMs
+ (NSDictionary<NSString *, id> *)macStatus;
/// What Documents/env.txt and Documents/metal changed at launch ("" if nothing).
+ (NSString *)envNote;
@end

NS_ASSUME_NONNULL_END
