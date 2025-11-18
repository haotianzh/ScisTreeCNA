import cupy as cp


kernel_log_probability = r"""
extern "C" __global__
void compute_genotype_log_probs(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    if (copy < 0 || copy > 2 * CN_MAX) return;

    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            if (g0 + g1 != copy) continue;
            if (g0 == 0 && g1 == 0) continue;

            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(copy + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
                        + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) continue;

                    float q = (float)g0_ / (g0_ + g1_);
                    float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                    float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

                    float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;

                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = prior + acc_log;

            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }

    float sumexp = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (log_probs[i] > NEG_INF) {
            sumexp += expf(log_probs[i] - maxval);
        }
    }

    float logZ = maxval + logf(fmaxf(sumexp, EPS));

    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i] - logZ;
    }
}
"""

# TODO: inlcuding 0 copy.
kernel_log_probability_cn_noise_with_zero_copy = r"""
extern "C" __global__ void compute_genotype_log_probs_cn_noise(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr, float cn_err,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    if (copy < 0 || copy > 2 * CN_MAX) return;

    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            // if (g0 + g1 < copy-1 || g0 + g1 > copy+1) continue;
            // if (g0 + g1 != copy) continue;
            // if (g0 == 0 && g1 == 0) continue;

            // poisson
            float log_cn_error = NEG_INF;
            log_cn_error = (g0 + g1) * logf(copy) - copy - lgammaf(g0 + g1 + 1.0f);
   
            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(g0 + g1 + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
                        + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) {
                        
                    
                    }else{
                        float q = (float)g0_ / (g0_ + g1_);
                        float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                        float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);
                        float pread = refc * logf(prob_ref) + altc * logf(prob_alt);
                    }

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;

                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = prior + acc_log;

            log_cn_error = logf(cn_err) + log_cn_error;
            if (g0 + g1 == copy){
                 log_cn_error = logf(expf(log_cn_error) + (1-cn_err)); 
            }
            log_probs[index] += log_cn_error;


            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }

    float sumexp = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (log_probs[i] > NEG_INF) {
            sumexp += expf(log_probs[i] - maxval);
        }
    }

    float logZ = maxval + logf(fmaxf(sumexp, EPS));

    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i] - logZ;
    }
}
"""

kernel_log_probability_cn_noise = r"""
extern "C" __global__ void compute_genotype_log_probs_cn_noise(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr, float cn_err,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            if (g0 == 0 && g1 == 0) continue;

            // poisson
            float log_cn_error = NEG_INF;
            // log_cn_error = (g0 + g1) * logf(copy) - copy - lgammaf(g0 + g1 + 1.0f);
            log_cn_error = copy * logf((g0 + g1)) - (g0 + g1) - lgammaf(copy + 1.0f);
            // printf("%f\n", log_cn_error);
            

            // normal
            // log_cn_error = -(copy - g0 - g1)*(copy - g0 - g1) / 2 / 0.01;

            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(g0 + g1 + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f);
                        // + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) continue;

                    float q = (float)g0_ / (g0_ + g1_);
                    float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                    float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

                    float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;
                
                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = acc_log;

            log_cn_error = logf(cn_err) + log_cn_error;
            if (g0 + g1 == copy){
                  log_cn_error = logf(expf(log_cn_error) + (1-cn_err)); 
            }
            if (copy != -1) log_probs[index] += log_cn_error;


            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }


    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i];
    }
}
"""

kernel_log_probability_cn_noise_original = r"""
extern "C" __global__ void compute_genotype_log_probs_cn_noise(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr, float cn_err,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    if (copy < 0 || copy > 2 * CN_MAX) return;

    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            // if (g0 + g1 < copy-1 || g0 + g1 > copy+1) continue;
            // if (g0 + g1 != copy) continue;
            if (g0 == 0 && g1 == 0) continue;

            // poisson
            float log_cn_error = NEG_INF;
            log_cn_error = (g0 + g1) * logf(copy) - copy - lgammaf(g0 + g1 + 1.0f);
   
            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(g0 + g1 + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
                        + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) continue;

                    float q = (float)g0_ / (g0_ + g1_);
                    float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                    float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

                    float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;

                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = prior + acc_log;

            log_cn_error = logf(cn_err) + log_cn_error;
            if (g0 + g1 == copy){
                 log_cn_error = logf(expf(log_cn_error) + (1-cn_err)); 
            }
            log_probs[index] += log_cn_error;


            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }

    float sumexp = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (log_probs[i] > NEG_INF) {
            sumexp += expf(log_probs[i] - maxval);
        }
    }

    float logZ = maxval + logf(fmaxf(sumexp, EPS));

    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i] - logZ;
    }
}
"""


kernel_log_probability_cn_noise2 = r"""
extern "C" __global__ void compute_genotype_log_probs_cn_noise(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr, float cn_err,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    if (copy < 0 || copy > 2 * CN_MAX) return;

    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            // if (g0 + g1 < copy-1 || g0 + g1 > copy+1) continue;
            // if (g0 + g1 != copy) continue;
            if (g0 == 0 && g1 == 0) continue;

            // poisson
            float log_cn_error = NEG_INF;
            log_cn_error = (g0 + g1) * logf(copy) - copy - lgammaf(g0 + g1 + 1.0f);
   
            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(g0 + g1 + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
                        + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) continue;

                    float q = (float)g0_ / (g0_ + g1_);
                    float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                    float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

                    float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;

                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = prior + acc_log + log_cn_error;

            if (g0 + g1 != copy){
                 log_probs[index] += logf(cn_err);
            }
            else{
                 log_probs[index] += logf(1- cn_err);
            }

            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }

    float sumexp = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (log_probs[i] > NEG_INF) {
            sumexp += expf(log_probs[i] - maxval);
        }
    }

    float logZ = maxval + logf(fmaxf(sumexp, EPS));

    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i] - logZ;
    }
}
"""


# mat1: (nsite, k) mat2: (k, k)
kernel_log_matmul = r"""
extern "C" __global__ void log_matmul(float* mat1, float* mat2, float* out, int n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < n && j < k) {
        out[i*k+j] = logf(0.0f); 
        for (int p=0; p<k; p++){
            float m = fmaxf(out[i*k+j], mat1[i*k+p] + mat2[j*k+p]);
            if (!__isinf(m))
                out[i*k+j] = m + logf(expf(out[i*k+j] - m) + expf(mat1[i*k+p] + mat2[j*k+p] - m));
        }   
    }
}
"""


# # non-contiguous pointers: logmatmul
# kernel_batch_log_matmul = r"""
# extern "C" __global__ void batch_log_matmul(
#     float** bmat1, float* mat2, float** bout, int batch, int n, int k)
# {
#     int z = blockIdx.z;
#     if (z >= batch) return;
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     int j = blockIdx.y * blockDim.y + threadIdx.y;
#     if (i >= n || j >= k) return;

#     float* mat1 = bmat1[z];
#     float* out  = bout[z];

#     float maxval = -1.0f/0.0f;
#     for (int p = 0; p < k; ++p) {
#         float val = mat1[i*k + p] + mat2[j*k + p];
#         if (val > maxval) maxval = val;
#     }
#     float sumexp = 0.0f;
#     for (int p = 0; p < k; ++p) {
#         float val = mat1[i*k + p] + mat2[j*k + p];
#         sumexp += expf(val - maxval);
#     }
#     out[i*k + j] = maxval + logf(sumexp);

#     if (i == 0 && j == 0 && z == 0){
#         printf("batch:%d, i:%i, j:%d, %f, %f\n", z, i, j, mat1[i*k+j], out[i*k+j]);
#         printf("%f %f %f \n", mat1[0], mat1[1], mat1[2]);
#     }

# }
# """


# non-contiguous pointers: logmatmul
kernel_batch_log_matmul = r"""
extern "C" __global__ void batch_log_matmul(float** bmat1, float* mat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z > m)
        return;
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float* mat1 = bmat1[z];
    float* out = bout[z]; 
    if (i < n && j < k) {
        out[i*k+j] = logf(0.0f); 
        for (int p=0; p<k; p++){
            float m = fmaxf(out[i*k+j], mat1[i*k+p] + mat2[j*k+p]);
            if (!isinf(m)){
                out[i*k+j] = m + logf(expf(out[i*k+j] - m) + expf(mat1[i*k+p] + mat2[j*k+p] - m));
            }
        } 
    }
}
"""

# non-contiguous pointers: logmatmul
kernel_batch_log_vecdot = r"""
extern "C" __global__ void batch_log_vecdot(float** bmat1, float** bmat2, float* bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z >= m) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float* mat1 = bmat1[z];
    float* mat2 = bmat2[z];

    float maxval = -1.0f/0.0f;
    for (int p = 0; p < k; ++p) {
        float val = mat1[i*k + p] + mat2[i*k + p];
        if (val > maxval) 
            maxval = val;
    }
    float sumexp = 0.0f;
    for (int p = 0; p < k; ++p) {
        float val = mat1[i*k + p] + mat2[i*k + p];
        sumexp += expf(val - maxval);
    }
    bout[z*n + i] = maxval + logf(sumexp);
}
"""


# non-contiguous pointers: logmatmul
kernel_batch_log_3vecdot = r"""
extern "C" __global__ void batch_log_3vecdot(float** bmat1, float** bmat2, float** bmat3, float* bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z >= m) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float* mat1 = bmat1[z];
    float* mat2 = bmat2[z];
    float* mat3 = bmat3[z];

    float maxval = -1.0f/0.0f;
    for (int p = 0; p < k; ++p) {
        float val = mat1[i*k + p] + mat2[i*k + p] + mat3[i*k + p];
        if (val > maxval) 
            maxval = val;
    }
    float sumexp = 0.0f;
    for (int p = 0; p < k; ++p) {
        float val = mat1[i*k + p] + mat2[i*k + p] + mat3[i*k + p];
        sumexp += expf(val - maxval);
    }
    bout[z*n + i] = maxval + logf(sumexp);
}
"""


# non-contiguous pointers: logmatadd
kernel_batch_matadd = r"""
extern "C" __global__ void batch_matadd(float** bmat1, float** bmat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z > m)
        return;
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float* mat1 = bmat1[z];
    float* mat2 = bmat2[z];
    float* out = bout[z]; 
    if (i < n && j < k) {
        out[i*k+j] = mat1[i*k+j] + mat2[i*k+j];
        //printf("%f\n", out[i*k+j]);
    }
}
"""

# non-contiguous pointers: logmatadd
kernel_batch_matadd_stride = r"""
extern "C" __global__ void batch_matadd_stride(float** bmat1, float** bmat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z > m)
        return;
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float* mat1 = bmat1[z];
    float* mat2 = bmat2[z];
    float* out = bout[z];
    if (i < n && j < k) {
        int l = i / k;
        out[i*k+j] = mat1[i*k+j] + mat2[l*k+j];
    }
}
"""


def compute_genotype_log_probs():
    return cp.RawKernel(kernel_log_probability, "compute_genotype_log_probs")


def compute_genotype_log_probs_cn_noise():
    return cp.RawKernel(
        kernel_log_probability_cn_noise, "compute_genotype_log_probs_cn_noise"
    )


def compute_genotype_log_probs_cn_noise_origin():
    return cp.RawKernel(
        kernel_log_probability_cn_noise_original, "compute_genotype_log_probs_cn_noise"
    )


def log_matmul_cuda():
    return cp.RawKernel(kernel_log_matmul, "log_matmul")


def batch_log_matmul_cuda():
    return cp.RawKernel(kernel_batch_log_matmul, "batch_log_matmul")


def batch_log_vecdot_cuda():
    return cp.RawKernel(kernel_batch_log_vecdot, "batch_log_vecdot")


def batch_log_3vecdot_cuda():
    return cp.RawKernel(kernel_batch_log_3vecdot, "batch_log_3vecdot")


def batch_matadd_cuda():
    return cp.RawKernel(kernel_batch_matadd, "batch_matadd")


def batch_matadd_stride_cuda():
    return cp.RawKernel(kernel_batch_matadd_stride, "batch_matadd_stride")


if __name__ == "__main__":
    # cp.random.seed(42)

    # a = cp.abs(cp.random.rand(64, 3))
    # # a = cp.array([[cp.e, cp.e, cp.e]], dtype=cp.float32)
    # b = cp.abs(cp.random.rand(3, 3))
    # # b = cp.array([[1e-3, 1e-3, 1-2e-3], [1e-3, 1e-3, 1-2e-3], [1e-3, 1e-3, 1-2e-3]], dtype=cp.float32)
    # # b = cp.array([])

    # a = cp.asarray(a, dtype=cp.float32)
    # a = a / cp.sum(a, axis=-1, keepdims=True)

    # b = cp.asarray(b, dtype=cp.float32)
    # # print(b)
    # b = b / cp.sum(b, axis=-1)

    # loga = cp.log(a)
    # logb = cp.log(b)

    # k = 1
    # # ------------ test cuda -------------

    # for i in range(k):
    #     c = cp.zeros([64, 3], dtype=cp.float32)
    #     # print(loga)
    #     log_matmul_cuda()((2, 1), (32, 32), (loga, logb, c, 64, 3))
    #     loga = c

    # # ------------ standard --------------
    # for i in range(k):
    #     a = cp.matmul(b, a.T).T

    # print(c)
    # print(cp.log(a))
    # print(cp.isclose(c, cp.log(a)))

    # -----------  test batch log matmul-----------------
    h = 700
    w = 35
    n = 10
    # for i in range(n):
    #     mats.append(cp.log(cp.random.rand(h, w).astype(cp.float32)))
    #     outs.append(cp.zeros((h, w), dtype=cp.float32))
    mats = [cp.log(cp.random.rand(h, w).astype(cp.float32)) for _ in range(n)]
    mat2 = cp.log(cp.eye(w).astype(cp.float32))
    outs = [cp.zeros((h, w), dtype=cp.float32) for _ in range(n)]
    bmat1 = cp.array([mat.data.ptr for mat in mats])
    bout = cp.array([mat.data.ptr for mat in outs])
    # mats = [cp.array([[-cp.inf, -cp.inf, 0.0]], dtype=cp.float32)]
    # # mats = [cp.array([[-cp.in, 2, 32]], dtype=cp.float32)]
    # outs = [cp.zeros([1,3], dtype=cp.float32)]
    # mat2 = cp.log(cp.eye(3, dtype=cp.float32))
    block_size = (16, 16)
    grid_size = (
        (h + block_size[0] - 1) // block_size[0],
        (w + block_size[1] - 1) // block_size[1],
        n,
    )
    # print(mats[0])
    # print(mat2)
    # print(cp.exp(mats[0]) @ cp.exp(mat2).T)
    batch_log_matmul_cuda()(grid_size, block_size, (bmat1, mat2, bout, n, h, w))
    for mat, out in zip(mats, outs):
        cpu_mm = cp.log(cp.matmul(cp.exp(mat), cp.exp(mat2).T))
        print(cp.allclose(cpu_mm, out, atol=1e-5))

    for mat, out in zip(mats, outs):
        mm = cp.log(cp.matmul(cp.exp(mat), cp.exp(mat2).T))
        # print(mm.shape)
        # print(out.shape)
        # print(mm)
        # print(out)
        print(cp.allclose(mm, out, atol=1e-4))
        # print(cp.abs(mm - out).sum())
        # break

    # ------------------ test batch log matmul2 --------------
    # Generate random matrices
    batch = 10
    n = 700
    k = 35

    mats1 = [cp.log(cp.random.rand(n, k).astype(cp.float32)) for _ in range(batch)]
    mat2 = cp.log(cp.eye(k).astype(cp.float32))
    outs = [cp.zeros((n, k), dtype=cp.float32) for _ in range(batch)]

    # Prepare pointers
    bmat1 = cp.array([mat.data.ptr for mat in mats1])
    bout = cp.array([mat.data.ptr for mat in outs])

    # Launch kernel
    block = (16, 16)
    grid = ((n + block[0] - 1) // block[0], (k + block[1] - 1) // block[1], batch)
    batch_log_matmul_cuda()(grid, block, (bmat1, mat2, bout, batch, n, k))
    for mat, out in zip(mats1, outs):
        cpu_mm = cp.log(cp.matmul(cp.exp(mat), cp.exp(mat2).T))
        print("allclose:", cp.allclose(cpu_mm, out, atol=1e-5))
        print("Max abs diff:", cp.abs(cpu_mm - out).max())

    # -------------- test batch add ------------------
    # mat1s = []
    # mat2s = []
    # outs = []
    # for i in range(10):
    #     mat1s.append(cp.log(cp.random.rand(64, 3)).astype(cp.float32))
    #     mat2s.append(cp.log(cp.random.rand(64, 3)).astype(cp.float32))
    #     outs.append(cp.zeros([64, 3], dtype=cp.float32))
    # batch_matadd_cuda()((5, 1, 10), (32, 32), (cp.array([v.data.ptr for v in mat1s]),
    #                                             cp.array([v.data.ptr for v in mat2s]),
    #                                             cp.array([v.data.ptr for v in outs]),
    #                                             10, 64, 3))
    # for mat1, mat2, out in zip(mat1s, mat2s, outs):
    #     mm = mat1 + mat2
    #     print(out)
    #     print(cp.isclose(mm, out, rtol=1e-4))
    #     # break

    ## --------- test matadd stride -------------
    # mat1s = []
    # mat2s = []
    # outs = []
    # aa = cp.log(cp.random.rand(2, 9, 3)).astype(cp.float32)
    # bb = cp.log(cp.random.rand(2, 3, 3)).astype(cp.float32)
    # for i in range(2):
    #     # mat1s.append(cp.log(cp.random.rand(9, 3)).astype(cp.float32))
    #     # mat2s.append(cp.log(cp.random.rand(3, 3)).astype(cp.float32))
    #     # outs.append(cp.log(cp.random.rand(9, 3)).astype(cp.float32))
    #     mat1s.append(aa[i])
    #     mat2s.append(bb[i])
    # # print(mat1s[0])
    # # print(mat2s[0])
    # batch_matadd_stride_cuda()((1, 1, 2), (32, 32), (cp.array([v.data.ptr for v in mat1s]),
    #                                             cp.array([v.data.ptr for v in mat2s]),
    #                                             cp.array([v.data.ptr for v in mat1s]),
    #                                             2, 9, 3))
    # print(mat1s[0])

    # ## ---------------- test vecdot ------------
    # mat1s = [cp.log(cp.random.rand(32, 10)).astype(cp.float32) for _ in range(10)]
    # mat2s = [cp.log(cp.random.rand(32, 10)).astype(cp.float32) for _ in range(10)]
    # out = cp.zeros([10, 32], dtype=cp.float32)
    # block_size = (32, 1)
    # grid_size = (10, 1, 10)
    # batch_log_vecdot_cuda()(grid_size, block_size, (cp.array([v.data.ptr for v in mat1s]),
    #                                                 cp.array([v.data.ptr for v in mat2s]),
    #                                                 out,
    #                                                 10,
    #                                                 32,
    #                                                 10))
    # print(out[0])

    # for mat1, mat2 in zip(mat1s, mat2s):
    #     mat1 = cp.exp(mat1)
    #     mat2 = cp.exp(mat2)
    #     res = cp.log((mat1 * mat2).sum(axis=-1))
    #     print(res)
    #     break
