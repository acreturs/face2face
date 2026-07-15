// =============================================================================
// CUDA photometric inner solver — the GPU counterpart of the per-pixel Ceres
// solve inside CeresFitter::fitPhotometric (the ~95% hot spot).
//
// The correspondence set is FIXED for the inner solve (fixed-visibility
// differentiable rendering), so this is an embarrassingly parallel problem:
// for each covered pixel the residual is
//
//   r = sqrtWeight · ( targetColour − inputImage( project(R·S + t) ) )         (RGB)
//
//   S = basePoint + Σ_k bBasis[k]·shape[k]     (aligned, barycentric-blended
//                                               surface point; the shape term is
//                                               only present when shape is free —
//                                               otherwise frozen shape is folded
//                                               into basePoint on the host)
//
// We build the Gauss–Newton normal equations (JᵀJ, Jᵀr) on the GPU using CENTRAL
// FINITE DIFFERENCES for the Jacobian (robust to write correctly — the residual
// is the single source of truth; no hand-derived rotation/projection/image-grad
// chain). The tiny nParams×nParams system is solved on the host (Eigen) in a
// Levenberg–Marquardt loop, calling cudaPhotoNormalEq / cudaPhotoCost here.
//
// Differences from the CPU/Ceres path (documented in GPU_RENDERER.md):
//   • image sampling is BILINEAR here vs Ceres' BiCubic — sub-pixel differences;
//   • Jacobian is finite-difference vs autodiff — direction matches to O(h²).
// Both are fine for tracking convergence; the CPU path remains the reference.
//
// Pure CUDA — no Eigen/OpenCV. Talks to CeresFitter.cpp via the extern "C" API.
// =============================================================================
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            std::fprintf(stderr, "[cuda-photo] %s:%d: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(_e));                               \
    } while (0)

static constexpr int MAXK  = 180;  // must be >= kShapeCoefficientCount (CeresFitter.h)
static constexpr int MAXNP = 186;  // 6 pose + MAXK shape

// Central-difference steps per parameter type. Small enough to stay within a
// texel of the bilinear image (so the gradient is local), large enough to avoid
// float cancellation. Tunable — see GPU_RENDERER.md.
static constexpr double H_AA    = 2e-3;   // angle-axis (rad)
static constexpr double H_T     = 0.2;    // translation (mm)
static constexpr double H_SHAPE = 1e-2;   // shape coeff (unitless)

struct CudaPhotoState {
    int   H = 0, W = 0, P = 0, K = 0;      // K = shape count folded into device
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int   optimizeShape = 0;

    float*  d_image = nullptr;             // 3*H*W  RGB [0,1]
    double* d_base  = nullptr;             // 3*P
    double* d_basis = nullptr;             // K*3*P  (null if !optimizeShape)
    double* d_tgt   = nullptr;             // 3*P

    double* d_params = nullptr;            // 3 aa + 3 t + K shape
    double* d_JtJ    = nullptr;            // MAXNP*MAXNP
    double* d_Jtr    = nullptr;            // MAXNP
    double* d_cost   = nullptr;            // 1
};

// ── device helpers ───────────────────────────────────────────────────────────

// Ceres-compatible angle-axis rotation (Rodrigues, with small-angle branch).
__device__ __forceinline__ void aaRotate(const double* aa, const double* p, double* out)
{
    const double t2 = aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2];
    if (t2 > 1e-24) {
        const double theta = sqrt(t2);
        const double c = cos(theta), s = sin(theta);
        const double k0 = aa[0] / theta, k1 = aa[1] / theta, k2 = aa[2] / theta;
        const double kp = k0 * p[0] + k1 * p[1] + k2 * p[2];
        const double cx = k1 * p[2] - k2 * p[1];
        const double cy = k2 * p[0] - k0 * p[2];
        const double cz = k0 * p[1] - k1 * p[0];
        out[0] = p[0] * c + cx * s + k0 * kp * (1.0 - c);
        out[1] = p[1] * c + cy * s + k1 * kp * (1.0 - c);
        out[2] = p[2] * c + cz * s + k2 * kp * (1.0 - c);
    } else {                                       // out = p + aa × p
        out[0] = p[0] + (aa[1] * p[2] - aa[2] * p[1]);
        out[1] = p[1] + (aa[2] * p[0] - aa[0] * p[2]);
        out[2] = p[2] + (aa[0] * p[1] - aa[1] * p[0]);
    }
}

// Bilinear sample with border clamp (Ceres' Grid2D clamps out-of-range too).
__device__ __forceinline__ void sampleBilinear(const float* img, int H, int W,
                                                double u, double v, double* out)
{
    if (u < 0.0) u = 0.0; if (u > W - 1.0) u = W - 1.0;
    if (v < 0.0) v = 0.0; if (v > H - 1.0) v = H - 1.0;
    const int x0 = (int)floor(u), y0 = (int)floor(v);
    const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    const double ax = u - x0, ay = v - y0;
    for (int c = 0; c < 3; ++c) {
        const double c00 = img[(y0 * W + x0) * 3 + c];
        const double c10 = img[(y0 * W + x1) * 3 + c];
        const double c01 = img[(y1 * W + x0) * 3 + c];
        const double c11 = img[(y1 * W + x1) * 3 + c];
        const double top = c00 * (1.0 - ax) + c10 * ax;
        const double bot = c01 * (1.0 - ax) + c11 * ax;
        out[c] = top * (1.0 - ay) + bot * ay;
    }
}

// Bilinear sample + analytic image gradient (dI/du, dI/dv) with border clamp.
// Used by the analytic-Jacobian path. Gradient is the exact derivative of the
// bilinear interpolant (piecewise-constant across texels); at a clamped border
// the corresponding gradient component is 0.
__device__ __forceinline__ void sampleBilinearGrad(const float* img, int H, int W,
                                                    double u, double v,
                                                    double* out, double* dIdu, double* dIdv)
{
    if (u < 0.0) u = 0.0; if (u > W - 1.0) u = W - 1.0;
    if (v < 0.0) v = 0.0; if (v > H - 1.0) v = H - 1.0;
    const int x0 = (int)floor(u), y0 = (int)floor(v);
    const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    const double ax = u - x0, ay = v - y0;
    for (int c = 0; c < 3; ++c) {
        const double c00 = img[(y0 * W + x0) * 3 + c];
        const double c10 = img[(y0 * W + x1) * 3 + c];
        const double c01 = img[(y1 * W + x0) * 3 + c];
        const double c11 = img[(y1 * W + x1) * 3 + c];
        const double top = c00 * (1.0 - ax) + c10 * ax;
        const double bot = c01 * (1.0 - ax) + c11 * ax;
        out[c]  = top * (1.0 - ay) + bot * ay;
        dIdu[c] = (1.0 - ay) * (c10 - c00) + ay * (c11 - c01);
        dIdv[c] = (1.0 - ax) * (c01 - c00) + ax * (c11 - c10);
    }
}

// One pixel's residual (RGB) at parameters (aa, t, shape).
__device__ __forceinline__ void pixelResidual(
    const double* aa, const double* t, const double* shape, int K,
    const double* base, const double* basisPix,        // basisPix: K*3 or null
    const double* tgt, double sqrtW,
    const float* img, int H, int W, float fx, float fy, float cx, float cy,
    double* r)
{
    double S[3] = { base[0], base[1], base[2] };
    for (int k = 0; k < K; ++k) {
        S[0] += basisPix[3 * k + 0] * shape[k];
        S[1] += basisPix[3 * k + 1] * shape[k];
        S[2] += basisPix[3 * k + 2] * shape[k];
    }
    double rot[3];
    aaRotate(aa, S, rot);
    const double X = rot[0] + t[0], Y = rot[1] + t[1], Z = rot[2] + t[2];
    const double u = fx * X / Z + cx;
    const double v = fy * Y / Z + cy;
    double obs[3];
    sampleBilinear(img, H, W, u, v, obs);
    r[0] = sqrtW * (tgt[0] - obs[0]);
    r[1] = sqrtW * (tgt[1] - obs[1]);
    r[2] = sqrtW * (tgt[2] - obs[2]);
}

// Map an active-parameter index to its perturbation. Layout (matching the host):
//   [0..5]  pose (aa0,aa1,aa2,t0,t1,t2)  when optimizePose
//   [..]    shape 0..K-1                 when optimizeShape
__device__ __forceinline__ double stepFor(int a, int poseParams)
{
    if (a < poseParams) return (a < 3) ? H_AA : H_T;
    return H_SHAPE;
}

// Kernel: accumulate JᵀJ, Jᵀr and Huber cost over all pixels.
// analytic==0 → finite-difference Jacobian (perturb the global angle-axis).
// analytic==1 → analytic Jacobian (image-gradient × projection × pose/shape
//               chain); pose rotation is a LOCAL perturbation δ (∂cam/∂δ =
//               −[R·S]_×), which the host composes as R(δ)·R_cur.
__global__ void normalEqKernel(const CudaPhotoState s, int optimizePose, int analytic,
                               double sqrtW, double huberDelta,
                               double* JtJ, double* Jtr, double* cost)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.P) return;

    const int poseParams = optimizePose ? 6 : 0;
    const int nP = poseParams + (s.optimizeShape ? s.K : 0);

    double aa[3]    = { s.d_params[0], s.d_params[1], s.d_params[2] };
    double t[3]     = { s.d_params[3], s.d_params[4], s.d_params[5] };
    double shape[MAXK];
    for (int k = 0; k < s.K; ++k) shape[k] = s.d_params[6 + k];

    const double* base     = &s.d_base[3 * p];
    const double* basisPix = s.optimizeShape ? &s.d_basis[3 * s.K * p] : nullptr;
    const double* tgt      = &s.d_tgt[3 * p];
    const int     Keff     = s.optimizeShape ? s.K : 0;

    double r0[3];
    pixelResidual(aa, t, shape, Keff, base, basisPix, tgt, sqrtW,
                  s.d_image, s.H, s.W, s.fx, s.fy, s.cx, s.cy, r0);

    // Huber reweighting on the 3-vector block: s2 = ‖r0‖².
    const double s2  = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2];
    const double d2  = huberDelta * huberDelta;
    const double w   = (s2 <= d2) ? 1.0 : sqrt(huberDelta / sqrt(s2));  // sqrt(ρ'(s))
    const double rho = (s2 <= d2) ? s2 : (2.0 * huberDelta * sqrt(s2) - d2);
    atomicAdd(cost, rho);

    double J[3 * MAXNP];
    if (analytic) {
        // Recompute geometry at the current params (linearisation point).
        double S[3] = { base[0], base[1], base[2] };
        for (int k = 0; k < Keff; ++k) {
            S[0] += basisPix[3 * k + 0] * shape[k];
            S[1] += basisPix[3 * k + 1] * shape[k];
            S[2] += basisPix[3 * k + 2] * shape[k];
        }
        double rot[3];
        aaRotate(aa, S, rot);
        const double X = rot[0] + t[0], Y = rot[1] + t[1], Z = rot[2] + t[2];
        const double u = s.fx * X / Z + s.cx, v = s.fy * Y / Z + s.cy;

        double val[3], dIdu[3], dIdv[3];
        sampleBilinearGrad(s.d_image, s.H, s.W, u, v, val, dIdu, dIdv);

        // ∂u/∂cam = (fx/Z, 0, −fx·X/Z²);  ∂v/∂cam = (0, fy/Z, −fy·Y/Z²)
        const double duX = s.fx / Z, duZ = -s.fx * X / (Z * Z);
        const double dvY = s.fy / Z, dvZ = -s.fy * Y / (Z * Z);
        const double wsq = -w * sqrtW;   // Huber weight × residual scale × (−1)

        int a = 0;
        if (optimizePose) {
            // rotation δ (local): columns of ∂cam/∂δ = −[rot]_×
            const double dc[3][3] = {
                {  0.0,     -rot[2],  rot[1] },
                {  rot[2],   0.0,    -rot[0] },
                { -rot[1],   rot[0],  0.0    } };
            for (int j = 0; j < 3; ++j) {
                const double du_dp = duX * dc[j][0] + duZ * dc[j][2];
                const double dv_dp = dvY * dc[j][1] + dvZ * dc[j][2];
                for (int c = 0; c < 3; ++c)
                    J[3 * a + c] = wsq * (dIdu[c] * du_dp + dIdv[c] * dv_dp);
                ++a;
            }
            // translation: ∂cam/∂t_j = e_j
            for (int j = 0; j < 3; ++j) {
                const double du_dp = (j == 0 ? duX : 0.0) + (j == 2 ? duZ : 0.0);
                const double dv_dp = (j == 1 ? dvY : 0.0) + (j == 2 ? dvZ : 0.0);
                for (int c = 0; c < 3; ++c)
                    J[3 * a + c] = wsq * (dIdu[c] * du_dp + dIdv[c] * dv_dp);
                ++a;
            }
        }
        if (s.optimizeShape) {
            for (int k = 0; k < s.K; ++k) {
                double bk[3] = { basisPix[3 * k + 0], basisPix[3 * k + 1], basisPix[3 * k + 2] };
                double rb[3];
                aaRotate(aa, bk, rb);                     // ∂cam/∂shape_k = R_cur·basis_k
                const double du_dp = duX * rb[0] + duZ * rb[2];
                const double dv_dp = dvY * rb[1] + dvZ * rb[2];
                for (int c = 0; c < 3; ++c)
                    J[3 * a + c] = wsq * (dIdu[c] * du_dp + dIdv[c] * dv_dp);
                ++a;
            }
        }
    } else {
        // Finite-difference Jacobian (central differences on the global params).
        for (int a = 0; a < nP; ++a) {
            double aap[3] = { aa[0], aa[1], aa[2] };
            double tp[3]  = { t[0], t[1], t[2] };
            double shp[MAXK];
            for (int k = 0; k < s.K; ++k) shp[k] = shape[k];

            const double h = stepFor(a, poseParams);
            double* slot;
            if (a < poseParams) slot = (a < 3) ? &aap[a] : &tp[a - 3];
            else                slot = &shp[a - poseParams];

            const double save = *slot;
            *slot = save + h;
            double rp[3];
            pixelResidual(aap, tp, shp, Keff, base, basisPix, tgt, sqrtW,
                          s.d_image, s.H, s.W, s.fx, s.fy, s.cx, s.cy, rp);
            *slot = save - h;
            double rm[3];
            pixelResidual(aap, tp, shp, Keff, base, basisPix, tgt, sqrtW,
                          s.d_image, s.H, s.W, s.fx, s.fy, s.cx, s.cy, rm);
            *slot = save;

            const double inv2h = 1.0 / (2.0 * h);
            J[3 * a + 0] = w * (rp[0] - rm[0]) * inv2h;
            J[3 * a + 1] = w * (rp[1] - rm[1]) * inv2h;
            J[3 * a + 2] = w * (rp[2] - rm[2]) * inv2h;
        }
    }

    const double rw[3] = { w * r0[0], w * r0[1], w * r0[2] };
    for (int i = 0; i < nP; ++i) {
        double gi = J[3 * i + 0] * rw[0] + J[3 * i + 1] * rw[1] + J[3 * i + 2] * rw[2];
        atomicAdd(&Jtr[i], gi);
        for (int j = 0; j < nP; ++j) {
            const double hij = J[3 * i + 0] * J[3 * j + 0]
                             + J[3 * i + 1] * J[3 * j + 1]
                             + J[3 * i + 2] * J[3 * j + 2];
            atomicAdd(&JtJ[i * nP + j], hij);
        }
    }
}

// Kernel: Huber cost only (for LM trial evaluation — no Jacobian).
__global__ void costKernel(const CudaPhotoState s, double sqrtW,
                           double huberDelta, double* cost)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.P) return;

    double aa[3] = { s.d_params[0], s.d_params[1], s.d_params[2] };
    double t[3]  = { s.d_params[3], s.d_params[4], s.d_params[5] };
    double shape[MAXK];
    for (int k = 0; k < s.K; ++k) shape[k] = s.d_params[6 + k];

    const double* base     = &s.d_base[3 * p];
    const double* basisPix = s.optimizeShape ? &s.d_basis[3 * s.K * p] : nullptr;
    const double* tgt      = &s.d_tgt[3 * p];
    const int     Keff     = s.optimizeShape ? s.K : 0;

    double r0[3];
    pixelResidual(aa, t, shape, Keff, base, basisPix, tgt, sqrtW,
                  s.d_image, s.H, s.W, s.fx, s.fy, s.cx, s.cy, r0);
    const double s2 = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2];
    const double d2 = huberDelta * huberDelta;
    atomicAdd(cost, (s2 <= d2) ? s2 : (2.0 * huberDelta * sqrt(s2) - d2));
}

// ── extern "C" API (called from CeresFitter.cpp) ─────────────────────────────
extern "C" {

void* cudaPhotoPrepare(const float* image, int H, int W,
                       float fx, float fy, float cx, float cy,
                       int P, const double* basePoint, const double* bBasis,
                       const double* target, int K, int optimizeShape)
{
    CudaPhotoState* s = new CudaPhotoState();
    s->H = H; s->W = W; s->P = P; s->K = (optimizeShape ? K : 0);
    s->fx = fx; s->fy = fy; s->cx = cx; s->cy = cy;
    s->optimizeShape = optimizeShape;

    const size_t np = (size_t)H * W;
    CUDA_CHECK(cudaMalloc(&s->d_image, sizeof(float) * 3 * np));
    CUDA_CHECK(cudaMemcpy(s->d_image, image, sizeof(float) * 3 * np, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s->d_base, sizeof(double) * 3 * P));
    CUDA_CHECK(cudaMemcpy(s->d_base, basePoint, sizeof(double) * 3 * P, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&s->d_tgt, sizeof(double) * 3 * P));
    CUDA_CHECK(cudaMemcpy(s->d_tgt, target, sizeof(double) * 3 * P, cudaMemcpyHostToDevice));

    if (optimizeShape && bBasis) {
        CUDA_CHECK(cudaMalloc(&s->d_basis, sizeof(double) * (size_t)K * 3 * P));
        CUDA_CHECK(cudaMemcpy(s->d_basis, bBasis, sizeof(double) * (size_t)K * 3 * P, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&s->d_params, sizeof(double) * (6 + MAXK)));
    CUDA_CHECK(cudaMalloc(&s->d_JtJ, sizeof(double) * MAXNP * MAXNP));
    CUDA_CHECK(cudaMalloc(&s->d_Jtr, sizeof(double) * MAXNP));
    CUDA_CHECK(cudaMalloc(&s->d_cost, sizeof(double)));
    return s;
}

// Uploads (aa,t,shape) → params, runs normalEqKernel, downloads JtJ/Jtr/cost.
void cudaPhotoNormalEq(void* handle, const double* aa, const double* t, const double* shape,
                       int optimizePose, int optimizeShape, int analytic,
                       double sqrtWeight, double huberDelta,
                       double* JtJ, double* Jtr, double* cost)
{
    CudaPhotoState* s = (CudaPhotoState*)handle;
    (void)optimizeShape;  // baked into the handle at prepare()
    double params[6 + MAXK] = {0};
    params[0] = aa[0]; params[1] = aa[1]; params[2] = aa[2];
    params[3] = t[0];  params[4] = t[1];  params[5] = t[2];
    for (int k = 0; k < s->K; ++k) params[6 + k] = shape[k];
    CUDA_CHECK(cudaMemcpy(s->d_params, params, sizeof(double) * (6 + s->K), cudaMemcpyHostToDevice));

    const int poseParams = optimizePose ? 6 : 0;
    const int nP = poseParams + (s->optimizeShape ? s->K : 0);
    CUDA_CHECK(cudaMemset(s->d_JtJ, 0, sizeof(double) * nP * nP));
    CUDA_CHECK(cudaMemset(s->d_Jtr, 0, sizeof(double) * nP));
    CUDA_CHECK(cudaMemset(s->d_cost, 0, sizeof(double)));

    const int TPB = 128;
    normalEqKernel<<<(s->P + TPB - 1) / TPB, TPB>>>(*s, optimizePose, analytic, sqrtWeight,
                                                    huberDelta, s->d_JtJ, s->d_Jtr, s->d_cost);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(JtJ, s->d_JtJ, sizeof(double) * nP * nP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Jtr, s->d_Jtr, sizeof(double) * nP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cost, s->d_cost, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

double cudaPhotoCost(void* handle, const double* aa, const double* t, const double* shape,
                     int optimizePose, int optimizeShape,
                     double sqrtWeight, double huberDelta)
{
    CudaPhotoState* s = (CudaPhotoState*)handle;
    (void)optimizePose; (void)optimizeShape;
    double params[6 + MAXK] = {0};
    params[0] = aa[0]; params[1] = aa[1]; params[2] = aa[2];
    params[3] = t[0];  params[4] = t[1];  params[5] = t[2];
    for (int k = 0; k < s->K; ++k) params[6 + k] = shape[k];
    CUDA_CHECK(cudaMemcpy(s->d_params, params, sizeof(double) * (6 + s->K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(s->d_cost, 0, sizeof(double)));

    const int TPB = 128;
    costKernel<<<(s->P + TPB - 1) / TPB, TPB>>>(*s, sqrtWeight, huberDelta, s->d_cost);
    CUDA_CHECK(cudaGetLastError());
    double cost = 0.0;
    CUDA_CHECK(cudaMemcpy(&cost, s->d_cost, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return cost;
}

void cudaPhotoDestroy(void* handle)
{
    CudaPhotoState* s = (CudaPhotoState*)handle;
    if (!s) return;
    cudaFree(s->d_image); cudaFree(s->d_base); cudaFree(s->d_tgt);
    if (s->d_basis) cudaFree(s->d_basis);
    cudaFree(s->d_params); cudaFree(s->d_JtJ); cudaFree(s->d_Jtr); cudaFree(s->d_cost);
    delete s;
}

}  // extern "C"
