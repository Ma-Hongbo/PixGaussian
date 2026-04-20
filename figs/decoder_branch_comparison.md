# Decoder Branch Comparison

This note summarizes the decoder path used by the `main` and `gaussian` branches for the current PixNerD class-conditional setup.

## Main Branch Decoder

Source: [PixGaussian_main/src/models/transformer/pixnerd_c2i.py](/home/hongbo/PixGaussian_main/src/models/transformer/pixnerd_c2i.py:251)

Current config dimensions:
- `s`: `1024`
- `x`: `64`
- decoder blocks: `2`
- patch size: `16`

```mermaid
flowchart LR
    S["Patch state s<br/>[B*L, 1024]"] --> H1["NerfBlock #1<br/>Hypernetwork: 1024 -> 16384<br/>split fc1/fc2, normalize dynamic weights"]
    X0["Patch pixels x<br/>[B*L, 16*16, 3]"] --> XE["NerfEmbedder<br/>(3 + DCT64) -> 64"]
    XE --> H1
    H1 --> H2["NerfBlock #2<br/>Per-patch dynamic MLP"]
    S --> H2
    H2 --> Q["NerfFinalLayer<br/>RMSNorm + Linear 64 -> 3"]
    Q --> F["Fold back to image"]
```

Decoder intuition:
- `main` keeps the decoder tiny in token width (`64`), but each patch gets its own dynamically generated MLP weights from `s`.
- The dynamic weights are normalized before the two matrix multiplications, so the decoder has both high per-patch capacity and stable scale control.

## Gaussian Branch Decoder

Source: [PixGaussian/src/models/transformer/pixnerd_c2i.py](/home/hongbo/PixGaussian/src/models/transformer/pixnerd_c2i.py:251)

Current config dimensions after the patch:
- `s`: `1024`
- `condition`: `1024`
- `x`: `64`
- shared point depth: `2`
- point MLP hidden: `512`
- conditioned decoder blocks: `2`
- patch size: `16`

```mermaid
flowchart LR
    S["Patch state s<br/>[B*L, 1024]"] --> C["PatchConditionHead<br/>RMSNorm + MLP<br/>1024 -> 1024"]
    X0["Patch pixels x<br/>[B*L, 16*16, 3]"] --> XE["NerfEmbedder<br/>(3 + DCT64) -> 64"]
    XE --> SP1["SharedPointEncoder #1<br/>Normed shared MLP<br/>64 -> 512 -> 64"]
    SP1 --> SP2["SharedPointEncoder #2<br/>Normed shared MLP<br/>64 -> 512 -> 64"]
    C --> CP1["ConditionedPointEncoder #1<br/>AdaNorm + gate<br/>64 -> 512 -> 64"]
    SP2 --> CP1
    C --> CP2["ConditionedPointEncoder #2<br/>AdaNorm + gate<br/>64 -> 512 -> 64"]
    CP1 --> CP2
    C --> Q["ConditionedQueryHead<br/>AdaNorm + Linear 64 -> 3"]
    CP2 --> Q
    Q --> F["Fold back to image"]
```

Decoder intuition:
- `gaussian` keeps the shared decoder idea: local point processing is shared across patches, while patch-specific information enters through a separate condition path.
- The patched version removes the severe `1024 -> 64` bottleneck, widens the point MLP to `512`, adds an extra shared point block, and restores normalization on the shared MLP weights plus the condition pathway.
- The conditioned projections are zero-initialized, so the decoder starts from a stable shared baseline and learns the patch-specific modulation gradually.
