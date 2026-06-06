#define _GNU_SOURCE
#include <time.h>

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "input.h"
#include "weights.h"
#include "q_input.h"
#include "q_weights.h"

#define D_MODEL 32
#define N_HEAD 8
#define HEAD_DIM (D_MODEL / N_HEAD)
#define NUM_CLASSES 5
#define EPS 1e-6f
#define T_CONV_MAX ((T_IN - K_CFG) / S_CFG + 1)
#define TCONV 40


#define KV_RSHIFT   3   
#define ATTN_RSHIFT 14  
#define PROJ_RSHIFT 0  

#define FINAL_ATTN_SCALE (SF_COMBINED * (float)(1u << (KV_RSHIFT + ATTN_RSHIFT + PROJ_RSHIFT)) / (float)(SCALE_FACTOR * SCALE_FACTOR))

static inline float ReLU(float x) { return x > 0.f ? x : 0.f; }

static inline float ftanhf(float x) {
    float y  = fminf(fmaxf(x, -4.0f), 4.0f);       // branchless clamp
    float y2 = y * y;
    float num = fmaf(y2, y, 27.0f * y);            // y*(27 + y^2)
    float den = fmaf(9.0f, y2, 27.0f);             // 27 + 9*y^2
    return num * (1.0f / den);                     // lets fast-math use rcp
}

static inline void DTanh(float *restrict x, size_t T,
                              const float *restrict alpha,
                              const float *restrict weight,
                              const float *restrict bias) {
    const float a = *alpha;
    for (size_t t = 0; t < T; ++t) {
        float *row = &x[t * D_MODEL];
        for (size_t d = 0; d < D_MODEL; ++d) {
            float v = tanhf(a * row[d]);
            row[d] = v * weight[d] + bias[d];
        }
    }
}

static inline void matmul(const float *restrict A, const float *restrict B,
                          float *restrict C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        const float *Ai = &A[i * k];
        float *Ci = &C[i * n];
        for (size_t j = 0; j < n; ++j) Ci[j] = 0.f;
        for (size_t p = 0; p < k; ++p) {
            const float ap = Ai[p];
            const float *Bp = &B[p * n];
            for (size_t j = 0; j < n; ++j) {
                Ci[j] += ap * Bp[j];
            }
        }
    }
}

static inline void add_bias(float *restrict X, size_t rows, size_t n,
                            const float *restrict bias) {
    for (size_t i = 0; i < rows; ++i) {
        float *row = &X[i * n];
        for (size_t j = 0; j < n; ++j) row[j] += bias[j];
    }
}

static inline void matmul_bias(const float *restrict A, const float *restrict B,
                               const float *restrict bias,
                               float *restrict C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        const float *Ai = &A[i * k];
        float *Ci = &C[i * n];
        for (size_t j = 0; j < n; ++j) Ci[j] = bias ? bias[j] : 0.f;
        for (size_t p = 0; p < k; ++p) {
            const float ap = Ai[p];
            const float *Bp = &B[p * n];
            for (size_t j = 0; j < n; ++j) {
                Ci[j] += ap * Bp[j];
            }
        }
    }
}

static inline void Linear_i8_i8(size_t in_features, size_t out_features,
                                const int8_t *restrict input,
                                const int8_t *restrict weight_kxn,
                                const int32_t *restrict bias,
                                int32_t *restrict output) {
    for (size_t j = 0; j < out_features; ++j) {
        int32_t s = bias ? bias[j] : 0;
        for (size_t i = 0; i < in_features; ++i)
            s += (int32_t)input[i] * (int32_t)weight_kxn[j * in_features + i];
        output[j] = s;
    }
}

static inline void Linear_f32(size_t in_features, size_t out_features,
                                 const float *restrict input,
                                 const float *restrict weight_kxn, 
                                 const float *restrict bias,       
                                 float *restrict output) {
    for (size_t j = 0; j < out_features; ++j) {
        float s = bias ? bias[j] : 0.f;
        for (size_t i = 0; i < in_features; ++i)
            s += input[i] * weight_kxn[i * out_features + j];
        output[j] = s;
    }
}

// Conv2d: input [2, T_in] -> output [D_MODEL, T_out]
static inline void Conv2d(const float *restrict x,
                          size_t k, size_t s,
                          float *restrict out,     // [D_MODEL][TCONV]
                          float *restrict seq) {   // [TCONV][D_MODEL] (transposed)
    for (size_t oc = 0; oc < D_MODEL; ++oc) {
        const float bias = conv_bias[oc];
        for (size_t ow = 0; ow < TCONV; ++ow) {
            float sum = bias;
            const size_t base_w = ow * s;
            for (size_t ic = 0; ic < 2; ++ic) {
                const size_t input_row  = ic * T_IN + base_w;
                const size_t weight_row = (oc * 2 + ic) * k;
                for (size_t kw = 0; kw < k; ++kw)
                    sum += x[input_row + kw] * conv_weight[weight_row + kw];
            }
            float v = ReLU(sum);
            seq[ow * D_MODEL + oc] = v;
        }
    }
}

static inline void QuantConv2d(const int16_t *restrict x,
                               size_t k, size_t s,
                               int32_t *restrict seq) { // [TCONV][D_MODEL]
    for (size_t oc = 0; oc < D_MODEL; ++oc) {
        const int32_t bias = q_conv_bias[oc];
        for (size_t ow = 0; ow < TCONV; ++ow) {
            int32_t sum = bias;
            const size_t base_w = ow * s;
            for (size_t ic = 0; ic < 2; ++ic) {
                const size_t input_row  = ic * T_IN + base_w;
                const size_t weight_row = (oc * 2 + ic) * k;
                for (size_t kw = 0; kw < k; ++kw)
                    sum += (int32_t)x[input_row + kw] * (int32_t)q_conv_weight[weight_row + kw];
            }
            if (sum < 0) sum = 0;
            seq[ow * D_MODEL + oc] = sum;
        }
    }
}

// Multi-Head Linear Self Attention
static inline void LiTAN(const float *restrict x_in, size_t T,
                             const float *restrict w_qkv,
                             const float *restrict b_qkv,
                             const float *restrict w_out,
                             const float *restrict b_out,
                             float *restrict attn_out,
                             float *restrict buf_qkv,
                             float *restrict buf_q,
                             float *restrict buf_k,
                             float *restrict buf_v,
                             float *restrict buf_kv) {

    for (size_t t = 0; t < T; ++t) {
        Linear_f32(D_MODEL, 3 * D_MODEL,
                   &x_in[t * D_MODEL],
                   w_qkv, b_qkv,
                   &buf_qkv[t * 3 * D_MODEL]);
    }

    // split
    for (size_t t = 0; t < T; ++t) {
        const float *row = &buf_qkv[t * 3 * D_MODEL];
        for (size_t h = 0; h < N_HEAD; ++h) {
            const float *qsrc = &row[h * HEAD_DIM];
            const float *ksrc = &row[D_MODEL + h * HEAD_DIM];
            const float *vsrc = &row[2 * D_MODEL + h * HEAD_DIM];
            size_t base = h * T * HEAD_DIM + t * HEAD_DIM;
            memcpy(&buf_q[base], qsrc, HEAD_DIM * sizeof(float));
            memcpy(&buf_k[base], ksrc, HEAD_DIM * sizeof(float));
            memcpy(&buf_v[base], vsrc, HEAD_DIM * sizeof(float));
        }
    }

    // L1 norm q,k
    for (size_t h = 0; h < N_HEAD; ++h) {
        for (size_t t = 0; t < T; ++t) {
            float sumq = 0.f, sumk = 0.f;
            size_t base = h * T * HEAD_DIM + t * HEAD_DIM;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                sumq += fabsf(buf_q[base + d]);
                sumk += fabsf(buf_k[base + d]);
            }
            sumq += EPS; sumk += EPS;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                buf_q[base + d] /= sumq;
                buf_k[base + d] /= sumk;
            }
        }
    }

    // kv = k^T v
    for (size_t h = 0; h < N_HEAD; ++h) {
        float *kv = &buf_kv[h * HEAD_DIM * HEAD_DIM];
        for (size_t i = 0; i < HEAD_DIM; ++i) {
            for (size_t j = 0; j < HEAD_DIM; ++j) kv[i * HEAD_DIM + j] = 0.f;
        }
        for (size_t t = 0; t < T; ++t) {
            const float *krow = &buf_k[h * T * HEAD_DIM + t * HEAD_DIM];
            const float *vrow = &buf_v[h * T * HEAD_DIM + t * HEAD_DIM];
            for (size_t i = 0; i < HEAD_DIM; ++i) {
                const float ki = krow[i];
                float *kv_row = &kv[i * HEAD_DIM];
                for (size_t j = 0; j < HEAD_DIM; ++j) {
                    kv_row[j] += ki * vrow[j];
                }
            }
        }
    }

    // q * kv
    for (size_t h = 0; h < N_HEAD; ++h) {
        const float *kv = &buf_kv[h * HEAD_DIM * HEAD_DIM];
        for (size_t t = 0; t < T; ++t) {
            float *vout = &buf_v[h * T * HEAD_DIM + t * HEAD_DIM];
            const float *qrow = &buf_q[h * T * HEAD_DIM + t * HEAD_DIM];
            for (size_t j = 0; j < HEAD_DIM; ++j) {
                float s = 0.f;
                for (size_t i = 0; i < HEAD_DIM; ++i) s += qrow[i] * kv[i * HEAD_DIM + j];
                vout[j] = s;
            }
        }
    }

    // concat heads
    for (size_t t = 0; t < T; ++t) {
        float *dst = &attn_out[t * D_MODEL];
        for (size_t h = 0; h < N_HEAD; ++h) {
            memcpy(&dst[h * HEAD_DIM], &buf_v[h * T * HEAD_DIM + t * HEAD_DIM], HEAD_DIM * sizeof(float));
        }
    }

    for (size_t t = 0; t < T; ++t) {
        Linear_f32(D_MODEL, D_MODEL,
                   &attn_out[t * D_MODEL],
                   w_out, b_out,
                   &buf_qkv[t * D_MODEL]);
    }
    memcpy(attn_out, buf_qkv, sizeof(float) * T * D_MODEL);
}

static inline void QuantLiTAN(const int8_t *restrict x_q, size_t T,
                              const int8_t *restrict w_qkv,
                              const int32_t *restrict b_qkv,
                              const int8_t *restrict w_out,
                              const int32_t *restrict b_out,
                              int32_t *restrict attn_out,               // [T][D_MODEL], integer output
                              int32_t *restrict buf_qkv,                // [T][3*D_MODEL]
                              int32_t *restrict buf_q,                  // [N_HEAD][T][HEAD_DIM]
                              int32_t *restrict buf_k,                  // [N_HEAD][T][HEAD_DIM]
                              int32_t *restrict buf_v,                  // [N_HEAD][T][HEAD_DIM]
                              int16_t *restrict buf_qn,                 // [N_HEAD][T][HEAD_DIM]
                              int16_t *restrict buf_kn,                 // [N_HEAD][T][HEAD_DIM]
                              int32_t *restrict buf_kv,                 // [N_HEAD][HEAD_DIM][HEAD_DIM]
                              int32_t *restrict buf_head_out,           // [N_HEAD][T][HEAD_DIM]
                              int32_t *restrict buf_proj) {             // [T][D_MODEL]
    const size_t qkv_cols = 3 * D_MODEL; // 96

    // 1) qkv projection — integer MACs, weight layout matches PyTorch (out-major)
    for (size_t t = 0; t < T; ++t) {
        const int8_t *in = &x_q[t * D_MODEL];
        int32_t *out = &buf_qkv[t * qkv_cols];
        for (size_t j = 0; j < qkv_cols; ++j) {
            int32_t s = b_qkv ? b_qkv[j] : 0;
            const size_t wrow = j * D_MODEL;
            for (size_t i = 0; i < D_MODEL; ++i)
                s += (int32_t)in[i] * (int32_t)w_qkv[wrow + i];
            out[j] = s;
        }
    }

    // 2) split
    for (size_t t = 0; t < T; ++t) {
        const int32_t *row = &buf_qkv[t * qkv_cols];
        for (size_t h = 0; h < N_HEAD; ++h) {
            size_t base = h * T * HEAD_DIM + t * HEAD_DIM;
            const size_t off_q = h * HEAD_DIM;
            const size_t off_k = D_MODEL + h * HEAD_DIM;
            const size_t off_v = 2 * D_MODEL + h * HEAD_DIM;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                buf_q[base + d] = row[off_q + d];
                buf_k[base + d] = row[off_k + d];
                buf_v[base + d] = row[off_v + d];
            }
        }
    }

    // 3) L1 norm to integer domain (float scaling only for normalization)
    for (size_t h = 0; h < N_HEAD; ++h) {
        for (size_t t = 0; t < T; ++t) {
            size_t base = h * T * HEAD_DIM + t * HEAD_DIM;
            float sumq = 0.f, sumk = 0.f;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                sumq += fabsf((float)buf_q[base + d]);
                sumk += fabsf((float)buf_k[base + d]);
            }
            sumq = sumq == 0.f ? 1.f : sumq;
            sumk = sumk == 0.f ? 1.f : sumk;
            const float scale_q = (float)SCALE_FACTOR / sumq;
            const float scale_k = (float)SCALE_FACTOR / sumk;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                int32_t q_scaled = (int32_t)lrintf((float)buf_q[base + d] * scale_q);
                int32_t k_scaled = (int32_t)lrintf((float)buf_k[base + d] * scale_k);
                if (q_scaled > 32767) q_scaled = 32767; else if (q_scaled < -32768) q_scaled = -32768;
                if (k_scaled > 32767) k_scaled = 32767; else if (k_scaled < -32768) k_scaled = -32768;
                buf_qn[base + d] = (int16_t)q_scaled;
                buf_kn[base + d] = (int16_t)k_scaled;
            }
        }
    }

    // 4) kv = k_norm^T * v (integer MACs with product downshift)
    for (size_t h = 0; h < N_HEAD; ++h) {
        int32_t *kv = &buf_kv[h * HEAD_DIM * HEAD_DIM];
        for (size_t i = 0; i < HEAD_DIM * HEAD_DIM; ++i) kv[i] = 0;
        for (size_t t = 0; t < T; ++t) {
            const int16_t *krow = &buf_kn[h * T * HEAD_DIM + t * HEAD_DIM];
            const int32_t *vrow = &buf_v[h * T * HEAD_DIM + t * HEAD_DIM];
            for (size_t i = 0; i < HEAD_DIM; ++i) {
                const int32_t ki = (int32_t)krow[i];
                int32_t *kv_row = &kv[i * HEAD_DIM];
                for (size_t j = 0; j < HEAD_DIM; ++j) {
                    int64_t prod = (int64_t)ki * (int64_t)vrow[j];
                    // round and shift to keep in int32
                    prod += (int64_t)1 << (KV_RSHIFT - 1);
                    kv_row[j] += (int32_t)(prod >> KV_RSHIFT);
                }
            }
        }
    }

    // 5) attn = q_norm * kv (per head) with product downshift
    for (size_t h = 0; h < N_HEAD; ++h) {
        const int32_t *kv = &buf_kv[h * HEAD_DIM * HEAD_DIM];
        for (size_t t = 0; t < T; ++t) {
            const int16_t *qrow = &buf_qn[h * T * HEAD_DIM + t * HEAD_DIM];
            int32_t *out_row = &buf_head_out[h * T * HEAD_DIM + t * HEAD_DIM];
            for (size_t j = 0; j < HEAD_DIM; ++j) {
                int32_t s = 0;
                for (size_t i = 0; i < HEAD_DIM; ++i) {
                    int64_t prod = (int64_t)qrow[i] * (int64_t)kv[i * HEAD_DIM + j];
                    prod += (int64_t)1 << (ATTN_RSHIFT - 1);
                    s += (int32_t)(prod >> ATTN_RSHIFT);
                }
                out_row[j] = s;
            }
        }
    }

    // 6) concat heads
    for (size_t t = 0; t < T; ++t) {
        int32_t *dst = &attn_out[t * D_MODEL];
        for (size_t h = 0; h < N_HEAD; ++h) {
            memcpy(&dst[h * HEAD_DIM], &buf_head_out[h * T * HEAD_DIM + t * HEAD_DIM], HEAD_DIM * sizeof(int32_t));
        }
    }

    // Optional additional downscale before final projection to protect accum
#if PROJ_RSHIFT > 0
    for (size_t i = 0; i < T * D_MODEL; ++i) {
        int64_t v = attn_out[i];
        v += (int64_t)1 << (PROJ_RSHIFT - 1);
        attn_out[i] = (int32_t)(v >> PROJ_RSHIFT);
    }
#endif

    // 7) final projection (integer MACs) — weight out-major
    for (size_t t = 0; t < T; ++t) {
        const int32_t *in_row = &attn_out[t * D_MODEL];
        int32_t *out_row = &buf_proj[t * D_MODEL];
        for (size_t j = 0; j < D_MODEL; ++j) {
            int32_t s = b_out ? b_out[j] : 0;
            const size_t wrow = j * D_MODEL;
            for (size_t i = 0; i < D_MODEL; ++i)
                s += in_row[i] * (int32_t)w_out[wrow + i];
            out_row[j] = s;
        }
        memcpy(&attn_out[t * D_MODEL], &buf_proj[t * D_MODEL], D_MODEL * sizeof(int32_t));
    }
}

// FFN
static inline void FFN(const float *restrict x, size_t T,
                       const float *restrict w1, const float *restrict b1,
                       const float *restrict w2, const float *restrict b2,
                       size_t dim_ffn,
                       float *restrict tmp, float *restrict out) {

    for (size_t t = 0; t < T; ++t) {
        Linear_f32(D_MODEL, dim_ffn,
                   &x[t * D_MODEL],
                   w1, b1,
                   &tmp[t * dim_ffn]);
        for (size_t i = 0; i < dim_ffn; ++i)
            tmp[t * dim_ffn + i] = ReLU(tmp[t * dim_ffn + i]);

        Linear_f32(dim_ffn, D_MODEL,
                   &tmp[t * dim_ffn],
                   w2, b2,
                   &out[t * D_MODEL]);
    }
}

static inline void AdaptiveAvgPool1d(const float *restrict x, size_t T,
                                       float *restrict y) {
    for (size_t d = 0; d < D_MODEL; ++d) {
        float s = 0.f;
        for (size_t t = 0; t < T; ++t) s += x[t * D_MODEL + d];
        y[d] = s / (float)T;
    }
}

void print_conv2d_output(const float *output, size_t T_out) {
    printf("Conv2d Output:\n");
    for (size_t d = 0; d < D_MODEL; ++d) {
        printf("Channel %zu: ", d);
        for (size_t t = 0; t < T_out; ++t) {
            printf("%f ", output[d * T_out + t]);
        }
        printf("\n\n");
    }
}

int main(void) {
    float logits[NUM_CLASSES];

    struct timespec start, end_conv, end_tf, end;
    struct timespec end_norm1, end_norm2, end_mhsa1, end_ffn1;

    // stage buffers (stack) sized by T_CONV_MAX
    int32_t seq_q_conv[T_CONV_MAX * D_MODEL];
    float seq[T_CONV_MAX * D_MODEL];
    float seq_norm[T_CONV_MAX * D_MODEL];

    // Integer attention buffers (integer MACs; float only for L1 scale)
    int8_t seq_q[T_CONV_MAX * D_MODEL];
    int32_t buf_qkv[T_CONV_MAX * 3 * D_MODEL];
    int32_t buf_q[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int32_t buf_k[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int32_t buf_v[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int16_t buf_qn[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int16_t buf_kn[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int32_t buf_kv[N_HEAD * HEAD_DIM * HEAD_DIM];
    int32_t buf_head[N_HEAD * T_CONV_MAX * HEAD_DIM];
    int32_t buf_proj[T_CONV_MAX * D_MODEL];
    int32_t attn_out_int[T_CONV_MAX * D_MODEL];
    float buf_tmp_ffn[T_CONV_MAX * DIM_FFN_CFG];
    float buf_out_ffn[T_CONV_MAX * D_MODEL];
    float attn_out[T_CONV_MAX * D_MODEL];

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Conv-Tokenizer
    Conv2d(input_signal, K_CFG, S_CFG, NULL, seq);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end_conv);

    // LiTAN - E-SpecFormer Encoder block 1
    memcpy(seq_norm, seq, sizeof(float) * TCONV * D_MODEL);
    DTanh(seq_norm, TCONV, &dt1_alpha[0], &dt1_weight[0][0], &dt1_bias[0][0]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_norm1);

    LiTAN(seq_norm, TCONV,
            qkv_weight[0], qkv_bias[0],
            outproj_weight[0], outproj_bias[0],
            attn_out, (float *)buf_qkv, (float *)buf_q, (float *)buf_k, (float *)buf_v, (float *)buf_kv);
    
    for (size_t i = 0; i < TCONV * D_MODEL; ++i) {
        seq[i] = seq_norm[i] + attn_out[i];
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_mhsa1);

    memcpy(seq_norm, seq, sizeof(float) * TCONV * D_MODEL);
    DTanh(seq_norm, TCONV, &dt2_alpha[0], &dt2_weight[0][0], &dt2_bias[0][0]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_norm2);

    FFN(seq_norm, TCONV,
        ffn1_weight[0], ffn1_bias[0],
        ffn2_weight[0], ffn2_bias[0],
        DIM_FFN_CFG, buf_tmp_ffn, buf_out_ffn);
    for (size_t i = 0; i < TCONV * D_MODEL; ++i) seq[i] += buf_out_ffn[i];
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_ffn1);

    // 3) Pool + FC
    float pooled[D_MODEL];
    AdaptiveAvgPool1d(seq, TCONV, pooled);
    for (size_t c = 0; c < NUM_CLASSES; ++c) {
        float s = fc_bias[c];
        const float *w = &fc_weight[c * D_MODEL];
        for (size_t d = 0; d < D_MODEL; ++d) s += pooled[d] * w[d];
        logits[c] = s;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Inference time: %f seconds\n", elapsed);

    double elapsed_conv = (end_conv.tv_sec - start.tv_sec) + 
                          (end_conv.tv_nsec - start.tv_nsec) / 1e9;
    printf("Conv2d time: %f seconds\n", elapsed_conv);

    double elapsed_norm11 = (end_norm1.tv_sec - end_conv.tv_sec) + 
                        (end_norm1.tv_nsec - end_conv.tv_nsec) / 1e9;
    printf("DTanh1 time: %f seconds\n", elapsed_norm11);  
    
    double elapsed_mhsa1 = (end_mhsa1.tv_sec - end_norm1.tv_sec) + 
                        (end_mhsa1.tv_nsec - end_norm1.tv_nsec) / 1e9;
    printf("LiTAN time: %f seconds\n", elapsed_mhsa1);  

    double elapsed_norm12 = (end_norm2.tv_sec - end_mhsa1.tv_sec) + 
                        (end_norm2.tv_nsec - end_mhsa1.tv_nsec) / 1e9;
    printf("DTan2 time: %f seconds\n", elapsed_norm12);  

    double elapsed_ffn1 = (end_ffn1.tv_sec - end_norm2.tv_sec) + 
                        (end_ffn1.tv_nsec - end_norm2.tv_nsec) / 1e9;
    printf("FFN time: %f seconds\n", elapsed_ffn1);  

    double elapsed_cls = (end.tv_sec - end_ffn1.tv_sec) +
                         (end.tv_nsec - end_ffn1.tv_nsec) / 1e9;

    printf("Pooled classifier time: %f seconds\n", elapsed_cls);  

    for (size_t i = 0; i < NUM_CLASSES; ++i) printf("%zu: %.3f\n", i, logits[i]);
    return 0;
}

