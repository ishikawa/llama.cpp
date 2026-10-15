# IQ2_S expert planar layout (fork extension)

`GGML_TYPE_IQ2_S_EP` (43) stores the exact IQ2_S bytes in a different order. It is intended for Metal MoE gate/up weights. There is no CPU decoder, dot product, or quantizer. CPU computation rejects it, including `--cpu-moe` placement. Only Metal `MUL_MAT_ID` and the existing gate/up fusion of two such operations are supported; `MUL_MAT`, `GET_ROWS`, and conversion operations reject it.

For `nblk = ne[0]/256` and `blocks = ne[1]*nblk`, each expert contains:

```
payload[ne[1]][nblk][80]  // qs[64], qh[8], scales[8]
d[ne[1]][nblk]            // original half bytes
```

The block size is 256 elements and the nominal block size is 82 bytes, so all `nb[]` and tensor sizes remain unchanged. Row views do not describe independent rows of this representation. Expert views and copies must retain complete experts; the Metal support check requires contiguous weights. The loader's `check_tensors` validates only the d planes of complete experts.

The payload plane size is always a multiple of 16 and is checked by the converter and tensor validation. Given a 16-byte aligned tensor base, all expert bases and d planes are 16-byte aligned if `blocks % 8 == 0` (`82*blocks % 16 == 0`). This holds for `(2560,640,512)`: each expert has 6400 blocks, 512000 payload bytes and 524800 total bytes. These shapes use aligned `uint4` payload loads. Other complete block/row shapes use byte loads when the expert stride cannot guarantee alignment, without padding or a change to strides. The half plane needs only 2-byte alignment for its scalar loads.

The fused EP kernels always use nr0=2 and nsg=4. `GGML_METAL_MOE_GATE_UP_IQ2_S_EP=1` (also the default when unset) selects `kernel_moe_gate_up_iq2_s_ep`: each lane handles the original 32-element slices, reuses its 32 activation values across two rows, and loads qs/signs with one uint each per slice. The small-ubatch MV kernel uses the same dot. `GGML_METAL_MOE_GATE_UP_IQ2_S_EP=2` selects `kernel_moe_gate_up_iq2_s_ep_v2`: each lane processes 128 elements with one uint4 each for qs/signs, one uint each for qh/scales, and a separate half d. The input vector is shared in threadgroup memory after one cooperative load and barrier (10240 bytes for 2560 elements). If that vector exceeds the device's threadgroup memory limit, dispatch falls back to variant 1. All variants keep the original dot's final factor of 0.25. Fusion requires complete eight-row tiles; other row counts use unfused `MUL_MAT_ID`. IQ2_S decode, TG, NR0 and NSG environment overrides do not select EP variants. Models with no EP tensors retain their existing paths.

The same environment variable selects these additional fused candidates (no extra threadgroup memory):

| Value | Kernel | Work per lane | Arithmetic |
| --- | --- | --- | --- |
| 3 | `kernel_moe_gate_up_iq2_s_ep_lanes16` | 16 elements | Float dot, `yl[16]` reused across two rows |
| 4 | `kernel_moe_gate_up_iq2_s_ep_half` | 32 elements | Half4 dots, each dot converted to float before accumulation |
| 5 | `kernel_moe_gate_up_iq2_s_ep_lanes16_half` | 16 elements | Same half4 dots with the 16-element lane mapping |

For variants 3 and 5, 2560 input elements produce 160 units per row: five iterations with all 32 lanes active. Each unit reads two grid indices and two sign bytes from the payload, selects the correct qh and scale nibbles, and reads the original half d from its plane. Variant 3 retains float activations; variants 4 and 5 convert activations to half4 once per iteration and reuse them across gate/up and both rows. Grid entries are small integers represented exactly in half. Signs are applied by XOR of half sign bits using ushort4 masks, without per-element `select`. Half4 dot results are converted separately to float; scales, accumulation, the final factor of 0.25 and SwiGLU remain float. These are compute candidates; their speed and Metal correctness must be measured before choosing a winner.


## Conversion

Run from the repository root, with NumPy installed:

```sh
python3 gguf-py/gguf/scripts/gguf_iq2_s_ep.py /path/model-00001-of-00005.gguf /path/new-output-dir --verify
```

The tool processes all shards with their original names and layout. Only IQ2_S tensors matching `ffn_gate_exps` or `ffn_up_exps` are changed. It copies all bytes first, then patches those payloads and their tensor type IDs. Metadata, shard layout, offsets, other tensors and padding remain byte-identical. Memory use is bounded by an 8 MiB copy buffer and 8192-block chunks, plus the reader's memory mapping and metadata. Existing output files are rejected. A failed run can leave incomplete output files; remove those files before retrying. `--verify` reads both files in chunks, unpacks every converted expert, and compares the original bytes.

Run the synthetic roundtrip and little/big endian GGUF preservation tests (CPU only):

```sh
PYTHONPATH=gguf-py python3 gguf-py/tests/test_iq2_s_ep.py
```

## Metal correctness (manual; requires GPU)

```sh
cmake --build build-decode-dev --config Release -j
build-decode-dev/bin/test-backend-ops --iq2-s-ep MTL0
```

The dedicated comparison mode builds IQ2_S and EP graphs on the same Metal backend, with identical deterministic random weights and activations. It uses `(2560,640,512)`, ten distinct selected experts per token, and token counts 1 (unfused, fused variants 1 through 5), 7, and 512. It compares both individual `MUL_MAT_ID` outputs and SwiGLU outputs in unfused cases; fused cases compare the final SwiGLU output. It also checks the 512-token F16 activation path. Each comparison rejects non-finite results and requires NMSE below `5e-7`, except fused half variants 4 and 5, which allow `5e-6`. The half threshold permits 0.224% relative RMS output error: activations and four-term dot results incur half rounding (unit roundoff about 0.049%), and the two matmul errors propagate through SwiGLU. This tolerance is limited to those two candidates; it is not a guarantee for arbitrary activation ranges. It needs approximately 1 GiB for the four weight tensors. Clear IQ2_S tuning environment overrides for the default comparison. The mode sets `GGML_METAL_FUSE_MOE_GATE_UP` and `GGML_METAL_MOE_GATE_UP_IQ2_S_EP` before initializing a backend for each case, so one invocation covers all five fused variants. The result lines report `ep_variant`.

Use `GGML_METAL_DISPATCH_STATS=1` to inspect kernel dispatches, including the fused kernel names above and `kernel_mul_mm_id_iq2_s_ep_f32`. Also run the existing regressions:

```sh
build-decode-dev/bin/test-backend-ops -b MTL0 -o MUL_MAT_ID
GGML_METAL_FUSE_MOE_GATE_UP=1 build-decode-dev/bin/test-backend-ops -b MTL0 -o MOE_GATE_UP_SWIGLU
```

These GPU checks and real-model conversion are deliberately not part of the CPU-only implementation validation.
