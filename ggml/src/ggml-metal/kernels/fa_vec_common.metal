constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];

//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];

constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];
constant int32_t FC_flash_attn_ext_vec_nh   [[function_constant(FC_FLASH_ATTN_EXT_VEC + 24)]];
constant bool    FC_flash_attn_ext_vec_has_sparse [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];
template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup

kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device const char * idx,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NH   (FC_flash_attn_ext_vec_nh)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*Q*C; // shared memory per simdgroup

    const int SMEM_Q = Q*NSG*PK;
    const int SMEM_S = NSG*SH;
    const int SMEM_O = 2*NSG*Q*PV;
    const int SMEM   = SMEM_Q + SMEM_S + SMEM_O;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    if (NH == 8 && DK == 256 && DV == 256 && NE == 1) {
        constexpr short NHG = 8;
        constexpr short SG  = 3*PV;

        const ushort iq3 = tgpig[2]/NWG;
        const ushort iq2 = tgpig[1]*NHG;
        const ushort iq1 = tgpig[0];

        threadgroup q4_t * sq4 = (threadgroup q4_t *) shmem_f16;
        threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + NHG*PK + sgitg*SG);
        threadgroup o4_t * sr4 = (threadgroup o4_t *) (shmem_f16 + NHG*PK + sgitg*SG + 2*PV);

        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;

        device const float4 * q4 = (device const float4 *) q;

        if (iq1 < args.ne01) {
            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[h*PK4 + i] = (q4_t) q4[h*(args.nb02/sizeof(float4)) + i];
                    } else {
                        sq4[h*PK4 + i] = (q4_t) 0.0f;
                    }
                }
            }
        }

        for (short i = tiisg; i < NHG*C; i += NW) {
            ss[i] = (s_t) 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S[NHG];
        float M[NHG];
        float slope[NHG];
        o4_t lo[NHG][DV4/NL];

        FOR_UNROLL (short h = 0; h < NHG; ++h) {
            S[h] = 0.0f;
            M[h] = -FLT_MAX/2;
            slope[h] = 1.0f;

            if (FC_flash_attn_ext_vec_has_bias) {
                const short hh = iq2 + h;

                const float base = hh < args.n_head_log2 ? args.m0 : args.m1;
                const short exph = hh < args.n_head_log2 ? hh + 1 : 2*(hh - args.n_head_log2) + 1;

                slope[h] = pow(base, exph);
            }

            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                lo[h][ii] = (o4_t) 0.0f;
            }
        }

        const short tx = tiisg;

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            const int ic_src = ic;
            device const char * kc = k;
            device const char * vc = v;
            device const char * mc = mask;

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                kc = pad;
                vc = kc + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mc = vc + args.nb21*C*args.ne_12_2*args.ne_12_3;

                kc += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                vc += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                ic = 0;
            }

            float mh[NHG];
            bool  active[NHG];
            bool any_active = false;

            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                if (FC_flash_attn_ext_vec_has_mask) {
                    device const half * pm;

                    if (FC_flash_attn_ext_vec_has_kvpad && ic_src + C > args.ne11) {
                        pm = (device const half *) mc +
                            iq1*C +
                            ((iq2 + h)%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    } else {
                        pm = (device const half *) (mc +
                            iq1*args.nb31 +
                            ((iq2 + h)%args.ne32)*args.nb32 +
                            (iq3%args.ne33)*args.nb33);
                    }

                    mh[h] = (float) pm[ic + tiisg];
                } else if (FC_flash_attn_ext_vec_has_kvpad && ic_src + tiisg >= args.ne11) {
                    mh[h] = -MAXHALF;
                } else {
                    mh[h] = 0.0f;
                }

                active[h] = simd_max(mh[h]) > -MAXHALF;
                any_active = any_active || active[h];
            }

            if (!any_active) {
                continue;
            }

            {
                if (is_same<kd4_t, k4_t>::value && is_same<q4_t, k4_t>::value) {
                    device      const half * pk = (device const half *) (kc + ic*args.nb11);
                    threadgroup const half * pq = (threadgroup const half *) sq4;

                    FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

                        simdgroup_half8x8 mk[2];
                        simdgroup_half8x8 mq[2];

                        #pragma unroll (MIN(DK/16, 16))
                        for (short ii = 0; ii < DK/16; ++ii) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*ii, PK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*ii, PK);

                            simdgroup_load(mk[0], pk + 8*cc*NS10 + 0*8 + 16*ii, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 8*cc*NS10 + 1*8 + 16*ii, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }

                        simdgroup_store(mqk, ss + 8*cc, C, 0, false);
                    }

                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    FOR_UNROLL (short h = 0; h < NHG; ++h) {
                        qk_t s = ss[h*C + tiisg]*args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            s = args.logit_softcap*precise::tanh(s);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            s += (qk_t) mh[h]*slope[h];
                        } else {
                            s += (qk_t) mh[h];
                        }

                        ss[h*C + tiisg] = active[h] ? s : (qk_t) -MAXHALF;
                    }
                } else {
                    device const k4_t * pk4 = (device const k4_t *) (kc + ic*args.nb11);
                    pk4 += tx;

                    FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                        qk_t mqk[NHG];

                        FOR_UNROLL (short h = 0; h < NHG; ++h) {
                            mqk[h] = 0.0f;
                        }

                        if (is_same<kd4_t, k4_t>::value) {
                            FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                const k4_t mk = pk4[cc*NS10/4 + ii*NL];

                                FOR_UNROLL (short h = 0; h < NHG; ++h) {
                                    mqk[h] += dot((float4) mk, (float4) sq4[h*PK4 + ii*NL + tx]);
                                }
                            }
                        } else {
                            device const kd4_t * pk = (device const kd4_t *) (kc + ((ic + cc)*args.nb11));

                            FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                const short i = ii*NL + tx;

                                k4_t mk;
                                deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                                FOR_UNROLL (short h = 0; h < NHG; ++h) {
                                    mqk[h] += dot((float4) mk, (float4) sq4[h*PK4 + i]);
                                }
                            }
                        }

                        FOR_UNROLL (short h = 0; h < NHG; ++h) {
                            mqk[h] = simd_sum(mqk[h]);

                            qk_t s = mqk[h]*args.scale;

                            if (FC_flash_attn_ext_vec_has_scap) {
                                s = args.logit_softcap*precise::tanh(s);
                            }

                            if (FC_flash_attn_ext_vec_has_bias) {
                                s += (qk_t) mh[h]*slope[h];
                            } else {
                                s += (qk_t) mh[h];
                            }

                            if (tiisg == cc) {
                                ss[h*C + cc] = active[h] ? s : (qk_t) -MAXHALF;
                            }
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                if (active[h]) {
                    const float m = M[h];
                    const float s = ss[h*C + tiisg];

                    M[h] = simd_max(max(M[h], s));

                    const float ms = exp(m - M[h]);
                    const float vs = exp(s - M[h]);

                    S[h] = S[h]*ms + simd_sum(vs);
                    ss[h*C + tiisg] = vs;

                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[h][ii] *= ms;
                    }
                } else {
                    ss[h*C + tiisg] = 0.0f;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (vc + ic*args.nb21);
                    pv4 += tx;

                    FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const float4 mv = (float4) pv4[cc*NS20/4 + ii*NL];

                            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                                lo[h][ii] += o4_t(mv*float4(ss[h*C + cc]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (vc + ((ic + cc)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                                lo[h][ii] += o4_t(float4(mv)*float4(ss[h*C + cc]));
                            }
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                const float m = M[h];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2 + h] : -FLT_MAX/2;

                M[h] = simd_max(max(M[h], s));

                const float ms = exp(m - M[h]);
                const float vs = exp(s - M[h]);

                S[h] = S[h]*ms + simd_sum(vs);

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[h][ii] *= ms;
                }
            }
        }

        for (short r = NSG/2; r > 0; r >>= 1) {
            if (sgitg < 2*r && tiisg == 0) {
                FOR_UNROLL (short h = 0; h < NHG; ++h) {
                    ss[2*h + 0] = S[h];
                    ss[2*h + 1] = M[h];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            float ms0[NHG];
            float ms1[NHG];

            if (sgitg < r) {
                threadgroup s_t * ssr = (threadgroup s_t *) (shmem_f16 + NHG*PK + (sgitg + r)*SG);

                FOR_UNROLL (short h = 0; h < NHG; ++h) {
                    const float S0 = S[h];
                    const float M0 = M[h];

                    const float S1 = ssr[2*h + 0];
                    const float M1 = ssr[2*h + 1];

                    const float Mv = max(M0, M1);

                    ms0[h] = exp(M0 - Mv);
                    ms1[h] = exp(M1 - Mv);

                    S[h] = S0*ms0[h] + S1*ms1[h];
                    M[h] = Mv;
                }
            }

            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (sgitg < 2*r) {
                        sr4[tiisg] = lo[h][ii];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (sgitg < r) {
                        threadgroup o4_t * srr4 = (threadgroup o4_t *) (shmem_f16 + NHG*PK + (sgitg + r)*SG + 2*PV);
                        lo[h][ii] = lo[h][ii]*ms0[h] + srr4[tiisg]*ms1[h];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        if (sgitg == 0) {
            const int64_t nrows = args.ne3*args.ne2*args.ne1;

            device float4 * dst4 = (device float4 *) dst;
            device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

            FOR_UNROLL (short h = 0; h < NHG; ++h) {
                const int64_t rid = iq3*args.ne2*args.ne1 + (iq2 + h) + iq1*args.ne1;
                const float Sinv = NWG == 1 ? (S[h] == 0.0f ? 0.0f : 1.0f/S[h]) : 1.0f;

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    const short i = ii*NL + tiisg;
                    dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) lo[h][ii]*Sinv;
                }

                if (NWG > 1 && tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = S[h];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = M[h];
                }
            }
        }

        return;
    }

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) shmem_f16; // holds the query data
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 + SMEM_Q + sgitg*SH); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 + SMEM_Q + sgitg*SH); // same as above but in s4_t
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 + SMEM_Q + sgitg*SH + 2*Q*C); // scratch buffer for mask
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + SMEM_Q + SMEM_S + 2*sgitg*Q*PV); // scratch buffer for the results

    // sparse indices for the current block
    threadgroup int * spidx = FC_flash_attn_ext_vec_has_sparse
        ? (threadgroup int *) (shmem_f16 + SMEM) + sgitg*C
        : nullptr;

    // store the result for all queries in shared memory (the O matrix from the paper)
    so4 += tiisg;

    {
        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load Q query rows to shared memory
    {
        for (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
            if (iq1_q < args.ne01) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[qq*PK4 + i] = (q4_t) q4[i];
                    } else {
                        sq4[qq*PK4 + i] = (q4_t) 0.0f;
                    }
                }
            } else {
                for (short i = tiisg; i < PK4; i += NW) {
                    sq4[qq*PK4 + i] = (q4_t) 0.0f;
                }
            }
        }
    }

    // zero out so
    for (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq*DV4 + i*NL] = (o4_t) 0.0f;
        }
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q];
        float M[Q];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            S[qq] = 0.0f;
            M[qq] = -FLT_MAX/2;
        }

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // pointer to the mask
        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        // sparse indices: the list of finite mask entries per query row
        // the sparse path requires Q == 1 (enforced by the host)
        device const int * pidx = nullptr;
        if (FC_flash_attn_ext_vec_has_sparse) {
            pidx = (device const int *) idx +
                ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded + (iq1%args.ne31)*args.n_kv_max_padded;
        }

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            device const half * pm[Q];
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                // padded query rows clamp to row 0 of the mask to avoid OOB; their scores
                // are forced to -inf below, so the values never affect the result.
                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (ic + tiisg >= args.ne11) {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        pm[qq] = (device const half *) (mask) +
                            (iq1*Q + qq)*C +
                            (iq2%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    }
                }

                ic = 0;
            }

            // load the sparse KV indices for the current block into shared memory
            if (FC_flash_attn_ext_vec_has_sparse) {
                FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                    const short i = ii*NW + tiisg;

                    spidx[i] = pidx[ic + i];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        const int i11 = spidx[tiisg];
                        if ((iq1*Q + qq) < args.ne01 && i11 >= 0) {
                            sm[qq*C + tiisg] = pm[qq][i11];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if ((iq1*Q + qq) < args.ne01) {
                            sm[qq*C + tiisg] = pm[qq][ic + tiisg];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                }
            } else {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) >= args.ne01) {
                        sm[qq*C + tiisg] = -MAXHALF;
                    }
                }
            }

            // skip -INF mask
            {
                bool any_finite = false;
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (simd_max(sm[qq*C + tiisg]) > -MAXHALF) {
                        any_finite = true;
                    }
                }
                if (!any_finite) {
                    continue;
                }
            }

            // Q*K^T
            {
                device      const k4_t * pk4 = nullptr;

                if (!FC_flash_attn_ext_vec_has_sparse) {
                    pk4 = (device const k4_t *) (k + ic*args.nb11);

                    pk4 += ty*NS10/4 + tx;
                }

                qk_t mqk[Q][C/NE];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        mqk[qq][cc] = 0.0f;
                    }
                }

                // each simdgroup processes Q queries and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (FC_flash_attn_ext_vec_has_sparse) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = spidx[NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<kd4_t, k4_t>::value) {
                                device const k4_t * pk4s = (device const k4_t *) (k + i11*args.nb11) + tx;
                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const k4_t k_elem = pk4s[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                                    }
                                }
                            } else {
                                device const kd4_t * pk = (device const kd4_t *) (k + i11*args.nb11);

                                k4_t mk;

                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                                    }
                                }
                            }
                        }
                    } else if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                            }
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                            }
                        }
                    }

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (NE == 1) {
                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                        } else {
                            // simdgroup reduce (NE = 4)
                            // [ 0 ..  7] -> [ 0]
                            // [ 8 .. 15] -> [ 8]
                            // [16 .. 23] -> [16]
                            // [24 .. 31] -> [24]
                            if (NE <= 1) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
                            }

                            // broadcast
                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (FC_flash_attn_ext_vec_has_mask &&
                       !FC_flash_attn_ext_vec_has_scap &&
                       !FC_flash_attn_ext_vec_has_bias) {
                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (qk_t) sm[qq*C + NE*tx + ty]);
                    } else {
                        mqk[qq][tx] *= args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            mqk[qq][tx] = args.logit_softcap*precise::tanh(mqk[qq][tx]);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty]*slope;
                        } else {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty];
                        }

                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const float m = M[qq];
                    const float s = ss[qq*C + tiisg];

                    M[qq] = simd_max(max(M[qq], s));

                    const float ms = exp(m - M[qq]);
                    const float vs = exp(s - M[qq]);

                    S[qq] = S[qq]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[qq*C + tiisg] = vs;

                    // O = diag(ms)*O
                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[Q][DV4/NL];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[qq][ii] = 0.0f;
                    }
                }

                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = spidx[NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<vd4_t, v4_t>::value) {
                                device const v4_t * pv4 = (device const v4_t *) (v + i11*args.nb21);

                                pv4 += tx;

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const v4_t v_elem = pv4[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            } else {
                                device const vd4_t * pv4 = (device const vd4_t *) (v + i11*args.nb21);

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    v4_t mv;

                                    deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            }
                        }
                    }
                } else if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + NE*cc + ty]));
                            }
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float m = M[qq];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[qq] = simd_max(max(M[qq], s));

                const float ms = exp(m - M[qq]);
                const float vs = exp(s - M[qq]);

                S[qq] = S[qq]*ms + simd_sum(vs);

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[qq*DV4 + ii*NL] *= ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                ss[2*qq + 0] = (s_t) S[qq];
                ss[2*qq + 1] = (s_t) M[qq];
            }
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;

            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            // interleave the workgroup data
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
            }

            // store S and M
            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NH
#undef NS10
#undef NS20
}
