#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif

constant uint GDN_R4D_C     = 32;
constant uint GDN_R4D_KD    = 128;
constant uint GDN_R4D_VD    = 128;
constant uint GDN_R4D_HK    = 16;
constant uint GDN_R4D_HV    = 48;
constant uint GDN_R4D_RATIO = 3;
constant uint GDN_R4D_BV    = 32;

kernel void kernel_gated_delta_net_r4d_kkt_c32_f32(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const float * k,
        device const float * g,
        device const float * beta,
        device float * inverse,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half staged_k[GDN_R4D_C * GDN_R4D_KD];
    threadgroup half staged_kt[GDN_R4D_KD * GDN_R4D_C];
    threadgroup float gram[GDN_R4D_C * GDN_R4D_C];

    const uint chunk = tgpig.x;
    const uint hk = tgpig.y;
    const uint seq = tgpig.z;
    const uint token0 = chunk * GDN_R4D_C;
    const uint nchunks = args.ne22 / GDN_R4D_C;

    for (uint i = tid; i < GDN_R4D_C * GDN_R4D_KD; i += 512) {
        const uint r = i / GDN_R4D_KD;
        const uint d = i - r * GDN_R4D_KD;
        const half value = half(k[((ulong(seq) * args.ne22 + token0 + r) * GDN_R4D_HK + hk) * GDN_R4D_KD + d]);
        staged_k[i] = value;
        staged_kt[d * GDN_R4D_C + r] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < 16) {
        const uint rt = sg / 4;
        const uint ct = sg - rt * 4;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint kb = 0; kb < GDN_R4D_KD / 8; ++kb) {
            simdgroup_half8x8 a;
            simdgroup_half8x8 b;
            simdgroup_load(a, staged_k + rt * 8 * GDN_R4D_KD + kb * 8, GDN_R4D_KD, 0, false);
            simdgroup_load(b, staged_kt + kb * 8 * GDN_R4D_C + ct * 8, GDN_R4D_C, 0, false);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, gram + rt * 8 * GDN_R4D_C + ct * 8, GDN_R4D_C, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < GDN_R4D_RATIO) {
        // Product GDN repeats the 16 Q/K heads in three 16-head tiles, matching i21 % ne11.
        const uint hv = hk + sg * GDN_R4D_HK;
        const uint column = lane;
        const device float * gate_ptr = g;
        const device float * beta_ptr = beta;
        const ulong seq_offset = ulong(seq) * args.ne22 * GDN_R4D_HV;
        float d[GDN_R4D_C];
        float gcs[GDN_R4D_C];
        float cumulative = 0.0f;
        for (uint i = 0; i < GDN_R4D_C; ++i) {
            const ulong gh = seq_offset + ulong(token0 + i) * GDN_R4D_HV + hv;
            cumulative += gate_ptr[gh];
            gcs[i] = cumulative;
            float sum = 0.0f;
            const float bi = beta_ptr[gh];
            for (uint j = 0; j < i; ++j) {
                const float lij = bi * exp(gcs[i] - gcs[j]) * gram[i * GDN_R4D_C + j];
                sum = fma(lij, d[j], sum);
            }
            d[i] = (i == column ? 1.0f : 0.0f) - sum;
            const ulong offset = (((ulong(seq) * nchunks + chunk) * GDN_R4D_HV + hv) * GDN_R4D_C + i) * GDN_R4D_C + column;
            inverse[offset] = d[i];
        }
    }
}

kernel void kernel_gated_delta_net_r4d_scan_c32_f32(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const float * q,
        device const float * k,
        device const float * v,
        device const float * inverse,
        device const float * g,
        device const float * beta,
        device const float * state_in,
        device float * output,
        device float * state_out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half shared[16320];
    threadgroup float chunk_g[GDN_R4D_C];
    threadgroup half * staged_k = shared;
    threadgroup half * staged_q = staged_k + GDN_R4D_C * GDN_R4D_KD;
    threadgroup half * staged_state = staged_q + GDN_R4D_C * GDN_R4D_KD;
    threadgroup half * temp = staged_state + GDN_R4D_BV * GDN_R4D_KD;
    threadgroup float * float_work = (threadgroup float *)staged_q;

    const uint slice = tgpig.x;
    const uint hv = tgpig.y;
    const uint seq = tgpig.z;
    const uint hk = hv % GDN_R4D_HK;
    const uint v0 = slice * GDN_R4D_BV;
    const uint mt = sg / 4;
    const uint vt = sg - mt * 4;
    const uint nchunks = args.ne22 / GDN_R4D_C;
    const ulong state_base = (ulong(seq) * GDN_R4D_HV + hv) * GDN_R4D_VD * GDN_R4D_KD + v0 * GDN_R4D_KD;
    const device float * initial_state = state_in;
    const device float * gate_ptr = g;
    const device float * beta_ptr = beta;
    const ulong seq_gate_offset = ulong(seq) * args.ne22 * GDN_R4D_HV;

    for (uint i = tid; i < GDN_R4D_BV * GDN_R4D_KD; i += 512) {
        const uint vr = i / GDN_R4D_KD;
        const uint d = i - vr * GDN_R4D_KD;
        staged_state[i] = half(initial_state[state_base + ulong(vr) * GDN_R4D_KD + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk = 0; chunk < nchunks; ++chunk) {
        const uint token0 = chunk * GDN_R4D_C;
        if (tid == 0) {
            float cumulative = 0.0f;
            for (uint r = 0; r < GDN_R4D_C; ++r) {
                cumulative += gate_ptr[seq_gate_offset + ulong(token0 + r) * GDN_R4D_HV + hv];
                chunk_g[r] = cumulative;
            }
        }
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_KD; i += 512) {
            const uint r = i / GDN_R4D_KD;
            const uint d = i - r * GDN_R4D_KD;
            staged_k[i] = half(k[((ulong(seq) * args.ne22 + token0 + r) * GDN_R4D_HK + hk) * GDN_R4D_KD + d]);
            staged_q[i] = half(q[((ulong(seq) * args.ne22 + token0 + r) * GDN_R4D_HK + hk) * GDN_R4D_KD + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 u = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 proj = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 out = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint kb = 0; kb < GDN_R4D_KD / 8; ++kb) {
            simdgroup_half8x8 km;
            simdgroup_half8x8 qm;
            simdgroup_half8x8 sm;
            simdgroup_half8x8 kj;
            simdgroup_load(km, staged_k + mt * 8 * GDN_R4D_KD + kb * 8, GDN_R4D_KD, 0, false);
            simdgroup_load(qm, staged_q + mt * 8 * GDN_R4D_KD + kb * 8, GDN_R4D_KD, 0, false);
            simdgroup_load(sm, staged_state + vt * 8 * GDN_R4D_KD + kb * 8, GDN_R4D_KD, 0, true);
            simdgroup_load(kj, staged_k + vt * 8 * GDN_R4D_KD + kb * 8, GDN_R4D_KD, 0, true);
            simdgroup_multiply_accumulate(u, km, sm, u);
            simdgroup_multiply_accumulate(out, qm, sm, out);
            simdgroup_multiply_accumulate(proj, qm, kj, proj);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_store(out, float_work + mt * 8 * GDN_R4D_BV + vt * 8, GDN_R4D_BV, 0, false);
        simdgroup_store(u, float_work + GDN_R4D_C * GDN_R4D_BV + mt * 8 * GDN_R4D_BV + vt * 8, GDN_R4D_BV, 0, false);
        simdgroup_store(proj, float_work + 2 * GDN_R4D_C * GDN_R4D_BV + mt * 8 * GDN_R4D_C + vt * 8, GDN_R4D_C, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float gl = chunk_g[GDN_R4D_C - 1];
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_BV; i += 512) {
            const uint tr = i / GDN_R4D_BV;
            const uint vc = i - tr * GDN_R4D_BV;
            const ulong gh = seq_gate_offset + ulong(token0 + tr) * GDN_R4D_HV + hv;
            const float vi = v[((ulong(seq) * args.ne22 + token0 + tr) * GDN_R4D_HV + hv) * GDN_R4D_VD + v0 + vc];
            const float ui = float_work[GDN_R4D_C * GDN_R4D_BV + i];
            float_work[GDN_R4D_C * GDN_R4D_BV + i] = beta_ptr[gh] * (vi - exp(chunk_g[tr]) * ui);

            const uint pr = i / GDN_R4D_C;
            const uint pc = i - pr * GDN_R4D_C;
            // Keep each decay factor <= 1 before narrowing to half. Splitting the exponent around
            // a chunk reference can overflow one WMMA operand even though their product is bounded.
            const float pv = pc <= pr ? rsqrt((float)GDN_R4D_KD) * exp(chunk_g[pr] - chunk_g[pc]) * float_work[2 * GDN_R4D_C * GDN_R4D_BV + i] : 0.0f;
            temp[GDN_R4D_C * GDN_R4D_BV + i] = half(pv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Keep (I + L)^-1 and its product with W in F32. Near-duplicate keys with beta close to
        // one can produce inverse entries outside the finite F16 range.
        const ulong inverse_base = ((ulong(seq) * nchunks + chunk) * GDN_R4D_HV + hv) * GDN_R4D_C * GDN_R4D_C;
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_BV; i += 512) {
            const uint tr = i / GDN_R4D_BV;
            const uint vc = i - tr * GDN_R4D_BV;
            float sum = 0.0f;
            for (uint j = 0; j < GDN_R4D_C; ++j) {
                sum = fma(inverse[inverse_base + tr * GDN_R4D_C + j],
                          float_work[GDN_R4D_C * GDN_R4D_BV + j * GDN_R4D_BV + vc], sum);
            }
            float_work[2 * GDN_R4D_C * GDN_R4D_BV + i] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_BV; i += 512) {
            temp[i] = half(float_work[2 * GDN_R4D_C * GDN_R4D_BV + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 correction = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint cb = 0; cb < GDN_R4D_C / 8; ++cb) {
            simdgroup_half8x8 sam;
            simdgroup_half8x8 vm;
            simdgroup_load(sam, temp + GDN_R4D_C * GDN_R4D_BV + mt * 8 * GDN_R4D_C + cb * 8, GDN_R4D_C, 0, false);
            simdgroup_load(vm, temp + cb * 8 * GDN_R4D_BV + vt * 8, GDN_R4D_BV, 0, false);
            simdgroup_multiply_accumulate(correction, sam, vm, correction);
        }
        simdgroup_store(correction, float_work + GDN_R4D_C * GDN_R4D_BV + mt * 8 * GDN_R4D_BV + vt * 8, GDN_R4D_BV, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_BV; i += 512) {
            const uint tr = i / GDN_R4D_BV;
            const uint vc = i - tr * GDN_R4D_BV;
            const float first = rsqrt((float)GDN_R4D_KD) * exp(chunk_g[tr]) * float_work[i];
            output[((ulong(seq) * args.ne22 + token0 + tr) * GDN_R4D_HV + hv) * GDN_R4D_VD + v0 + vc] = first + float_work[GDN_R4D_C * GDN_R4D_BV + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reuse the F16 delta tile for the state update after the attention correction consumed it.
        // gl <= chunk_g[tr] for the product's non-positive log gates, so this factor is bounded by 1.
        for (uint i = tid; i < GDN_R4D_C * GDN_R4D_BV; i += 512) {
            const uint tr = i / GDN_R4D_BV;
            temp[i] = half(exp(gl - chunk_g[tr]) * float(temp[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 delta0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 delta1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 delta2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 delta3 = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint tb = 0; tb < GDN_R4D_C / 8; ++tb) {
            simdgroup_half8x8 km;
            simdgroup_half8x8 vp0;
            simdgroup_half8x8 vp1;
            simdgroup_half8x8 vp2;
            simdgroup_half8x8 vp3;
            simdgroup_load(km, staged_k + tb * 8 * GDN_R4D_KD + sg * 8, GDN_R4D_KD, 0, false);
            simdgroup_load(vp0, temp + tb * 8 * GDN_R4D_BV + 0 * 8, GDN_R4D_BV, 0, true);
            simdgroup_load(vp1, temp + tb * 8 * GDN_R4D_BV + 1 * 8, GDN_R4D_BV, 0, true);
            simdgroup_load(vp2, temp + tb * 8 * GDN_R4D_BV + 2 * 8, GDN_R4D_BV, 0, true);
            simdgroup_load(vp3, temp + tb * 8 * GDN_R4D_BV + 3 * 8, GDN_R4D_BV, 0, true);
            simdgroup_multiply_accumulate(delta0, vp0, km, delta0);
            simdgroup_multiply_accumulate(delta1, vp1, km, delta1);
            simdgroup_multiply_accumulate(delta2, vp2, km, delta2);
            simdgroup_multiply_accumulate(delta3, vp3, km, delta3);
        }
        simdgroup_store(delta0, float_work + 0 * 8 * GDN_R4D_KD + sg * 8, GDN_R4D_KD, 0, false);
        simdgroup_store(delta1, float_work + 1 * 8 * GDN_R4D_KD + sg * 8, GDN_R4D_KD, 0, false);
        simdgroup_store(delta2, float_work + 2 * 8 * GDN_R4D_KD + sg * 8, GDN_R4D_KD, 0, false);
        simdgroup_store(delta3, float_work + 3 * 8 * GDN_R4D_KD + sg * 8, GDN_R4D_KD, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float state_decay = exp(gl);
        for (uint i = tid; i < GDN_R4D_BV * GDN_R4D_KD; i += 512) {
            const uint vr = i / GDN_R4D_KD;
            const uint d = i - vr * GDN_R4D_KD;
            const ulong index = state_base + ulong(vr) * GDN_R4D_KD + d;
            const float previous = chunk == 0 ? initial_state[index] : state_out[index];
            state_out[index] = float_work[i] + state_decay * previous;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (uint i = tid; i < GDN_R4D_BV * GDN_R4D_KD; i += 512) {
            const uint vr = i / GDN_R4D_KD;
            const uint d = i - vr * GDN_R4D_KD;
            staged_state[i] = half(state_out[state_base + ulong(vr) * GDN_R4D_KD + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
