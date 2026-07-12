// =============================================================================
// CUDA z-buffer rasteriser — the GPU counterpart of Renderer::rasterize +
// Renderer::interpolateShading (src/render/Renderer.cpp).
//
// This translation unit is PURE CUDA: no Eigen, no OpenCV. It talks to the rest
// of the program only through the three `extern "C"` launchers at the bottom,
// which take raw pointers + sizes. The Eigen/OpenCV-facing host code lives in
// CudaRenderer.cpp; nvcc compiles this file, g++ compiles that one.
//
// Algorithm (matches the CPU renderer's semantics exactly):
//   1. clearZ     — pack buffer set to "empty" (all-ones).
//   2. raster     — one thread per triangle; walk its 2D bbox, edge-function
//                   barycentrics, perspective-correct depth, then atomicMin a
//                   packed  (depthBits<<32 | triangleId)  key into the z-buffer.
//                   Packing means the SAME fragment wins as on the CPU: nearest
//                   depth first, ties broken by the smaller triangle index (the
//                   CPU keeps the first-written fragment on a strict-`<` test,
//                   i.e. the smaller f — identical rule).
//   3. resolve    — one thread per pixel; decode the winning triangle, recompute
//                   its perspective-correct barycentric weights at the pixel
//                   centre (same formulas as the CPU), blend the three shaded
//                   vertex colours, and write the G-buffer + image.
//
// Depth keys: camera-frame z is always > 0 here (we skip z<=1e-6), and for
// positive IEEE-754 floats the raw bit pattern is monotonic, so __float_as_uint
// gives a valid ordering for atomicMin.
// =============================================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            std::fprintf(stderr, "[cuda] %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(_e));                               \
    } while (0)

// Persistent device state — allocated once per renderer, reused every frame.
struct CudaRasterState {
    int H = 0, W = 0, numTris = 0, vertCap = 0;

    int*                d_tris   = nullptr;   // 3*numTris  (row-major i0,i1,i2)
    float*              d_vcam   = nullptr;   // 3*vertCap  camera-frame xyz
    float*              d_uv     = nullptr;   // 2*vertCap  projected pixels
    float*              d_shaded = nullptr;   // 3*vertCap  per-vertex RGB

    unsigned long long* d_zbuf   = nullptr;   // H*W  packed (depth<<32 | triId)

    float*              d_image  = nullptr;   // 3*H*W RGB
    float*              d_depth  = nullptr;   // H*W
    unsigned char*      d_mask   = nullptr;   // H*W
    int*                d_triIdx = nullptr;   // H*W
    float*              d_bary   = nullptr;   // 3*H*W
};

static constexpr unsigned long long kEmpty = 0xFFFFFFFFFFFFFFFFULL;

// ── kernel 1: clear the packed z-buffer ──────────────────────────────────────
__global__ void clearZKernel(unsigned long long* z, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = kEmpty;
}

// ── kernel 2: rasterise (one thread per triangle) ────────────────────────────
__global__ void rasterKernel(const int* tris, const float* vcam, const float* uv,
                             int numTris, int H, int W, unsigned long long* zbuf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numTris) return;

    const int i0 = tris[3 * f + 0], i1 = tris[3 * f + 1], i2 = tris[3 * f + 2];

    const float x0 = uv[2 * i0], y0 = uv[2 * i0 + 1];
    const float x1 = uv[2 * i1], y1 = uv[2 * i1 + 1];
    const float x2 = uv[2 * i2], y2 = uv[2 * i2 + 1];

    const float X0 = vcam[3 * i0], Y0 = vcam[3 * i0 + 1], z0 = vcam[3 * i0 + 2];
    const float X1 = vcam[3 * i1], Y1 = vcam[3 * i1 + 1], z1 = vcam[3 * i1 + 2];
    const float X2 = vcam[3 * i2], Y2 = vcam[3 * i2 + 1], z2 = vcam[3 * i2 + 2];

    // Back-face cull in the camera frame (matches Renderer::backfaceMask:
    // front-facing when the camera-space normal's z is negative).
    const float n_z = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
    if (!(n_z < 0.0f)) return;

    // Skip if any vertex is behind / on the camera plane (Renderer.cpp:106).
    if (z0 <= 1e-6f || z1 <= 1e-6f || z2 <= 1e-6f) return;

    int xmin = (int)floorf(fminf(x0, fminf(x1, x2)));
    int xmax = (int)ceilf (fmaxf(x0, fmaxf(x1, x2)));
    int ymin = (int)floorf(fminf(y0, fminf(y1, y2)));
    int ymax = (int)ceilf (fmaxf(y0, fmaxf(y1, y2)));
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > W - 1) xmax = W - 1;
    if (ymax > H - 1) ymax = H - 1;
    if (xmin > xmax || ymin > ymax) return;

    const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (fabsf(denom) < 1e-12f) return;
    const float invDenom = 1.0f / denom;

    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            const float px = x + 0.5f, py = y + 0.5f;

            const float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invDenom;
            const float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invDenom;
            const float w2 = 1.0f - w0 - w1;
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

            const float inv_z   = w0 / z0 + w1 / z1 + w2 / z2;
            const float z_pixel = 1.0f / inv_z;

            // Pack: nearest depth wins; ties → smaller triangle id (== CPU rule).
            const unsigned int      zb  = __float_as_uint(z_pixel);
            const unsigned long long key =
                ((unsigned long long)zb << 32) | (unsigned int)f;
            atomicMin(&zbuf[y * W + x], key);
        }
    }
}

// ── kernel 3: resolve winner → G-buffer + shaded image (one thread per pixel) ─
__global__ void resolveKernel(const int* tris, const float* vcam, const float* uv,
                              const float* shaded, int H, int W,
                              const unsigned long long* zbuf,
                              float* image, float* depth, unsigned char* mask,
                              int* triIdx, float* bary)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;

    const unsigned long long key = zbuf[p];
    if (key == kEmpty) {                              // no triangle here
        image[3 * p + 0] = 0.0f; image[3 * p + 1] = 0.0f; image[3 * p + 2] = 0.0f;
        depth[p]  = __int_as_float(0x7f800000);       // +inf (CPU empty value)
        mask[p]   = 0;
        triIdx[p] = -1;
        bary[3 * p + 0] = 0.0f; bary[3 * p + 1] = 0.0f; bary[3 * p + 2] = 0.0f;
        return;
    }

    const int          f       = (int)(key & 0xFFFFFFFFULL);
    const float        z_pixel  = __uint_as_float((unsigned int)(key >> 32));

    const int i0 = tris[3 * f + 0], i1 = tris[3 * f + 1], i2 = tris[3 * f + 2];
    const float x0 = uv[2 * i0], y0 = uv[2 * i0 + 1];
    const float x1 = uv[2 * i1], y1 = uv[2 * i1 + 1];
    const float x2 = uv[2 * i2], y2 = uv[2 * i2 + 1];
    const float z0 = vcam[3 * i0 + 2], z1 = vcam[3 * i1 + 2], z2 = vcam[3 * i2 + 2];

    const int   x  = p % W, y = p / W;
    const float px = x + 0.5f, py = y + 0.5f;

    const float denom    = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    const float invDenom = 1.0f / denom;
    const float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invDenom;
    const float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invDenom;
    const float w2 = 1.0f - w0 - w1;

    const float inv_z = w0 / z0 + w1 / z1 + w2 / z2;
    const float zp    = 1.0f / inv_z;
    const float b0 = w0 / z0 * zp, b1 = w1 / z1 * zp, b2 = w2 / z2 * zp;

    image[3 * p + 0] = b0 * shaded[3 * i0 + 0] + b1 * shaded[3 * i1 + 0] + b2 * shaded[3 * i2 + 0];
    image[3 * p + 1] = b0 * shaded[3 * i0 + 1] + b1 * shaded[3 * i1 + 1] + b2 * shaded[3 * i2 + 1];
    image[3 * p + 2] = b0 * shaded[3 * i0 + 2] + b1 * shaded[3 * i1 + 2] + b2 * shaded[3 * i2 + 2];

    depth[p]  = z_pixel;                              // == rasterise-time value
    mask[p]   = 255;
    triIdx[p] = f;
    bary[3 * p + 0] = b0; bary[3 * p + 1] = b1; bary[3 * p + 2] = b2;
}

// ── launchers (C ABI, called from CudaRenderer.cpp) ──────────────────────────
extern "C" {

void* cudaRasterCreate(int H, int W, const int* tris, int numTris)
{
    CudaRasterState* s = new CudaRasterState();
    s->H = H; s->W = W; s->numTris = numTris;
    const size_t np = (size_t)H * W;

    CUDA_CHECK(cudaMalloc(&s->d_tris, sizeof(int) * 3 * numTris));
    CUDA_CHECK(cudaMemcpy(s->d_tris, tris, sizeof(int) * 3 * numTris, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s->d_zbuf,   sizeof(unsigned long long) * np));
    CUDA_CHECK(cudaMalloc(&s->d_image,  sizeof(float) * 3 * np));
    CUDA_CHECK(cudaMalloc(&s->d_depth,  sizeof(float) * np));
    CUDA_CHECK(cudaMalloc(&s->d_mask,   sizeof(unsigned char) * np));
    CUDA_CHECK(cudaMalloc(&s->d_triIdx, sizeof(int) * np));
    CUDA_CHECK(cudaMalloc(&s->d_bary,   sizeof(float) * 3 * np));
    return s;
}

void cudaRasterRender(void* handle,
                      const float* vcam, const float* uv, const float* shaded,
                      int numVerts,
                      float* image, float* depth, unsigned char* mask,
                      int* triIdx, float* bary)
{
    CudaRasterState* s = (CudaRasterState*)handle;

    // (Re)allocate the per-vertex device buffers if the model grew.
    if (s->vertCap < numVerts) {
        if (s->d_vcam)   cudaFree(s->d_vcam);
        if (s->d_uv)     cudaFree(s->d_uv);
        if (s->d_shaded) cudaFree(s->d_shaded);
        CUDA_CHECK(cudaMalloc(&s->d_vcam,   sizeof(float) * 3 * numVerts));
        CUDA_CHECK(cudaMalloc(&s->d_uv,     sizeof(float) * 2 * numVerts));
        CUDA_CHECK(cudaMalloc(&s->d_shaded, sizeof(float) * 3 * numVerts));
        s->vertCap = numVerts;
    }

    CUDA_CHECK(cudaMemcpy(s->d_vcam,   vcam,   sizeof(float) * 3 * numVerts, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_uv,     uv,     sizeof(float) * 2 * numVerts, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_shaded, shaded, sizeof(float) * 3 * numVerts, cudaMemcpyHostToDevice));

    const int np  = s->H * s->W;
    const int TPB = 256;
    clearZKernel  <<<(np + TPB - 1) / TPB, TPB>>>(s->d_zbuf, np);
    rasterKernel  <<<(s->numTris + TPB - 1) / TPB, TPB>>>(
        s->d_tris, s->d_vcam, s->d_uv, s->numTris, s->H, s->W, s->d_zbuf);
    resolveKernel <<<(np + TPB - 1) / TPB, TPB>>>(
        s->d_tris, s->d_vcam, s->d_uv, s->d_shaded, s->H, s->W, s->d_zbuf,
        s->d_image, s->d_depth, s->d_mask, s->d_triIdx, s->d_bary);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(image,  s->d_image,  sizeof(float) * 3 * np, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(depth,  s->d_depth,  sizeof(float) * np,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mask,   s->d_mask,   sizeof(unsigned char) * np, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(triIdx, s->d_triIdx, sizeof(int) * np,       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bary,   s->d_bary,   sizeof(float) * 3 * np, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaRasterDestroy(void* handle)
{
    CudaRasterState* s = (CudaRasterState*)handle;
    if (!s) return;
    cudaFree(s->d_tris);   cudaFree(s->d_zbuf);
    cudaFree(s->d_image);  cudaFree(s->d_depth);  cudaFree(s->d_mask);
    cudaFree(s->d_triIdx); cudaFree(s->d_bary);
    if (s->d_vcam)   cudaFree(s->d_vcam);
    if (s->d_uv)     cudaFree(s->d_uv);
    if (s->d_shaded) cudaFree(s->d_shaded);
    delete s;
}

}  // extern "C"
