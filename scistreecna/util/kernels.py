import cupy as cp


# =============================================================================
# Genotype-likelihood kernels.
# Each computes, per (cell, site) pair (one CUDA thread = one tid = one cell-site),
# the log P(reads | genotype) for every genotype state and writes an N-vector to
# `out`. Everything is in LOG SPACE; NEG_INF (= log 0) is expected and normal.
#
# Genotype (g0, g1) = (#wild-type-base copies, #mutant-base copies). The per-state
# index is the lower-triangular packing of (g0+g1, g0):
#   index = (g0+g1)(g0+g1+1)/2 + g0 - CN_MIN(CN_MIN+1)/2
# Model: binomial reads with allelic dropout (ado) + sequencing error (seqerr),
# marginalized over surviving copies (g0_, g1_), plus a copy-number-noise term
# P(observed_cn | total) = (1-cn_err)*1[match] + cn_err*Poisson.
#
# Of the variants below, ONLY `kernel_log_probability_cn_noise` is LIVE (returned
# by the compute_genotype_log_probs_cn_noise() wrapper). The others
# (kernel_log_probability, *_with_zero_copy, *_original, *2) are ALTERNATE /
# COMMENTED-OUT versions kept for reference and are not used at runtime.
# =============================================================================

# ALTERNATE (not live): SNV-only genotype likelihood, no copy-number-noise term.
# Requires g0+g1 == observed copy number; normalizes each (cell,site) output to a
# log-probability distribution (subtracts logZ via two-pass log-sum-exp).
# In/out: ref/alt/cn (ncell*nsite), afs (nsite), out (ncell*nsite*N).
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

# ALTERNATE (not live): cn-noise variant that also keeps the 0-copy genotype.
# Adds the Poisson copy-number-noise term and normalizes to a distribution.
# NOTE: its inner read-model block is scoped inside an else{} so `pread` is local;
# kept for reference only.
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

# ===== LIVE kernel (used at runtime via compute_genotype_log_probs_cn_noise) =====
# Per (cell, site) thread, computes for every genotype (g0,g1) the UN-normalized
# log P(reads | g) + copy-number-noise term and writes the N-vector to `out`
# (NOT normalized to a distribution here, unlike the alternates).
# Inputs:
#   ref, alt : ref/alt read counts, length ncell*nsite (row-major site,cell)
#   cn       : observed total copy number per (cell,site); copy == -1 disables
#              the cn-noise term (missing/unknown CN)
#   afs      : per-site allele frequency (length nsite); currently the af prior
#              terms are commented out of `prior`
#   out      : output, length ncell*nsite*N (N genotype states per thread)
#   ado, seqerr, cn_err : allelic-dropout, sequencing-error, cn-noise rates
# Read model: marginalize over surviving copies (g0_,g1_) of (g0,g1); per surviving
# config, binomial mix prob q -> P(ref)/P(alt), accumulated via stable log-sum-exp.
# cn-noise: log_cn_error = cn_err * Poisson(copy; mean=g0+g1), with a (1-cn_err)
# point mass added when g0+g1 == copy (exact match).
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
                    // stable log-sum-exp accumulate: log(e^acc_log + e^val),
                    // pivoting on the larger term to avoid overflow.
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            // lower-triangular packing of (g0+g1, g0) into the flat state index.
            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = acc_log;

            // cn-noise mixture: cn_err * Poisson, plus (1-cn_err) point mass on exact match.
            log_cn_error = logf(cn_err) + log_cn_error;
            if (g0 + g1 == copy){
                  log_cn_error = logf(expf(log_cn_error) + (1-cn_err));
            }
            if (copy != -1) log_probs[index] += log_cn_error;  // copy==-1: no observed CN


            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }


    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i];
    }
}
"""

# ALTERNATE (not live): original cn-noise variant. Like the live one but keeps the
# allele-frequency prior (g0*log_af + g1*log_1maf) and normalizes each (cell,site)
# output to a log-probability distribution (subtracts logZ). Exposed via the
# compute_genotype_log_probs_cn_noise_origin() wrapper but not used at runtime.
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


# ALTERNATE (not live): cn-noise variant where the mixture is applied as a flat
# log(cn_err) / log(1-cn_err) weight (no Poisson+point-mass blend). Has no wrapper;
# kept for reference only.
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


# =============================================================================
# LOG-SPACE linear-algebra kernels for the tree-likelihood DP. All compute in log
# space: a log-domain "matmul" replaces sum-of-products with log-sum-exp of sums.
# =============================================================================

# Single log-domain matmul: out = log( exp(mat1) @ exp(mat2)^T ), i.e.
#   out[i,j] = logsumexp_p( mat1[i,p] + mat2[j,p] ).
# mat1: (n, k)  mat2: (k, k)  out: (n, k). Thread (i,j) = (row, col).
# mat2 is cooperatively staged into shared memory; a two-pass log-sum-exp
# (find max, then sum exp(.-max)) keeps it numerically stable. Empty rows -> -inf.
kernel_log_matmul = r"""
extern "C" __global__ void log_matmul(float* mat1, float* mat2, float* out, int n, int k){
    extern __shared__ float s_mat2[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    // Cooperatively load mat2 into shared memory
    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        // Two-pass log-sum-exp (numerically stable)
        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = mat1[i*k + p] + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = mat1[i*k + p] + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
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


# Batched log-domain matmul over m matrices (a single shared mat2).
# bmat1 / bout are device arrays of m POINTERS (one per batch element); batch index
# = blockIdx.z (z in [0,m)). Each mat1[z]/out[z] is (n, k); mat2 is (k, k) shared.
#   out[z][i,j] = logsumexp_p( mat1[z][i,p] + mat2[j,p] ).
# mat2 staged to shared memory once per block; two-pass log-sum-exp; empty -> -inf.
# (The commented block just above is the older non-shared-memory implementation.)
kernel_batch_log_matmul = r"""
extern "C" __global__ void batch_log_matmul(float** bmat1, float* mat2, float** bout, int m, int n, int k){
    extern __shared__ float s_mat2[];

    int z = blockIdx.z;
    if (z >= m) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    // Cooperatively load mat2 (k x k) into shared memory
    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        float* mat1 = bmat1[z];   // per-batch matrix pointer (double-pointer array)
        float* out = bout[z];

        // Two-pass log-sum-exp
        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = mat1[i*k + p] + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = mat1[i*k + p] + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
    }
}
"""

# Batched per-row log "dot product" of TWO matrices (elementwise sum then row reduce):
#   bout[z, i] = logsumexp_p( mat1[z][i,p] + mat2[z][i,p] ).
# bmat1/bmat2: arrays of m pointers; batch = blockIdx.z; mat[z] is (n, k); thread = row i.
# bout is a flat (m, n) array indexed bout[z*n + i]. One-pass log-sum-exp.
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


# Same as batch_log_vecdot but over THREE matrices:
#   bout[z, i] = logsumexp_p( mat1[z][i,p] + mat2[z][i,p] + mat3[z][i,p] ).
# bmat1/2/3: arrays of m pointers; batch = blockIdx.z; mat[z] is (n, k); thread = row i;
# bout flat (m, n) indexed bout[z*n + i].
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


# Batched elementwise add: out[z][i,j] = mat1[z][i,j] + mat2[z][i,j]
# (a plain add; in log space this is a log-domain elementwise product).
# bmat1/bmat2/bout: arrays of m pointers; batch = blockIdx.z; each mat[z] is (n, k);
# thread (i,j) = (row, col).
# non-contiguous pointers: logmatadd
kernel_batch_matadd = r"""
extern "C" __global__ void batch_matadd(float** bmat1, float** bmat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z >= m)
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

# Batched elementwise add with BROADCAST/stride on mat2: each k-row block of mat1
# is added to the same row of mat2. out[z][i,j] = mat1[z][i,j] + mat2[z][(i/k),j],
# i.e. mat2[z] (k, k) is broadcast across groups of k rows of mat1[z] (n, k).
# bmat1/bmat2/bout: arrays of m pointers; batch = blockIdx.z; thread (i,j).
# non-contiguous pointers: logmatadd
kernel_batch_matadd_stride = r"""
extern "C" __global__ void batch_matadd_stride(float** bmat1, float** bmat2, float** bout, int m, int n, int k){
    int z = blockIdx.z;
    if (z >= m)
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


# FUSED kernel: batch_matadd + batch_log_matmul in ONE launch, computing
# log( (A+B) @ mat2^T ) directly to save a kernel launch + a global round-trip of A+B.
#   out[z][i,j] = logsumexp_p( (mat_a[z][i,p] + mat_b[z][i,p]) + mat2[j,p] ).
# bmat_a/bmat_b/bout: arrays of m pointers; batch = blockIdx.z; mat[z] is (n, k);
# mat2 is (k, k) shared; thread (i,j); two-pass log-sum-exp; empty -> -inf.
# Computes: out[z] = log_matmul(mat_a[z] + mat_b[z], mat2)
kernel_batch_add_log_matmul = r"""
extern "C" __global__ void batch_add_log_matmul(
    float** bmat_a, float** bmat_b, float* mat2, float** bout,
    int m, int n, int k)
{
    extern __shared__ float s_mat2[];

    int z = blockIdx.z;
    if (z >= m) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_local = threadIdx.x * blockDim.y + threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;

    // Cooperatively load mat2 (k x k) into shared memory
    for (int idx = tid_local; idx < k * k; idx += total_threads) {
        s_mat2[idx] = mat2[idx];
    }
    __syncthreads();

    if (i < n && j < k) {
        float* mat_a = bmat_a[z];
        float* mat_b = bmat_b[z];
        float* out = bout[z];

        // Two-pass log-sum-exp over (mat_a + mat_b) @ mat2^T
        float maxval = -1.0f / 0.0f;
        for (int p = 0; p < k; ++p) {
            float val = (mat_a[i*k + p] + mat_b[i*k + p]) + s_mat2[j*k + p];
            if (val > maxval) maxval = val;
        }
        float sumexp = 0.0f;
        if (!isinf(maxval) || maxval > 0.0f) {
            for (int p = 0; p < k; ++p) {
                float val = (mat_a[i*k + p] + mat_b[i*k + p]) + s_mat2[j*k + p];
                if (!isinf(val)) sumexp += expf(val - maxval);
            }
        }
        out[i*k + j] = (sumexp > 0.0f) ? maxval + logf(sumexp) : -1.0f / 0.0f;
    }
}
"""


# Each function below compiles its kernel source into a cp.RawKernel and returns it.
# Call the returned kernel as kernel(grid, block, (args...)) (and shared_mem=... for
# the shared-memory kernels).

# Compile the ALTERNATE SNV-only genotype kernel (not used at runtime).
def compute_genotype_log_probs():
    return cp.RawKernel(kernel_log_probability, "compute_genotype_log_probs")


# Compile the LIVE genotype-likelihood-with-cn-noise kernel.
def compute_genotype_log_probs_cn_noise():
    return cp.RawKernel(
        kernel_log_probability_cn_noise, "compute_genotype_log_probs_cn_noise"
    )


# Compile the ALTERNATE "original" cn-noise kernel (af prior + normalization).
def compute_genotype_log_probs_cn_noise_origin():
    return cp.RawKernel(
        kernel_log_probability_cn_noise_original, "compute_genotype_log_probs_cn_noise"
    )


# Compile the single log-domain matmul kernel (needs shared_mem = k*k*4 bytes).
def log_matmul_cuda():
    return cp.RawKernel(kernel_log_matmul, "log_matmul")


# Compile the batched log-domain matmul kernel (needs shared_mem = k*k*4 bytes).
def batch_log_matmul_cuda():
    return cp.RawKernel(kernel_batch_log_matmul, "batch_log_matmul")


# Compile the batched 2-vector log-dot kernel.
def batch_log_vecdot_cuda():
    return cp.RawKernel(kernel_batch_log_vecdot, "batch_log_vecdot")


# Compile the batched 3-vector log-dot kernel.
def batch_log_3vecdot_cuda():
    return cp.RawKernel(kernel_batch_log_3vecdot, "batch_log_3vecdot")


# Compile the batched elementwise add kernel.
def batch_matadd_cuda():
    return cp.RawKernel(kernel_batch_matadd, "batch_matadd")


# Compile the batched elementwise add with row-broadcast (stride) kernel.
def batch_matadd_stride_cuda():
    return cp.RawKernel(kernel_batch_matadd_stride, "batch_matadd_stride")


# Compile the FUSED add+log-matmul kernel (needs shared_mem = k*k*4 bytes).
def batch_add_log_matmul_cuda():
    return cp.RawKernel(kernel_batch_add_log_matmul, "batch_add_log_matmul")


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
