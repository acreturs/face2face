#pragma once
#include <Eigen/Dense>
#include "Renderer.h"   // RenderInput / RenderOutput — the shared contract

// GPU (CUDA) drop-in for Renderer. Same construction signature and the same
// render() contract, so it substitutes the CPU Renderer anywhere a renderer is
// used as a value type (see runLiveImpl<> and runGpuCheck in src/main.cpp).
//
// Only the O(triangles × pixels) rasterisation + G-buffer resolve run on the
// GPU — that is the single-threaded hot loop in the CPU renderer. The cheap
// O(N) vertex stage (camera transform, per-vertex normals, SH shading) runs on
// the CPU using the *same* proj::/light::/Renderer::computeNormals helpers as
// Renderer, so those intermediate values are identical to the CPU path and the
// only place results can differ is float rounding inside the rasteriser (see
// GPU_RENDERER.md — verified with `--mode verify-gpu`).
//
// This header pulls in NO CUDA headers (the device handle is an opaque void*),
// so it is safe to include from ordinary g++ translation units. It is only
// referenced when the project is built with `make USE_CUDA=1`.
class CudaRenderer {
public:
    // Triangles + image size are constant for the whole run → set once. Uploads
    // the triangle list to the device and allocates the frame buffers.
    CudaRenderer(int height, int width, Eigen::MatrixX3i triangles);
    ~CudaRenderer();

    // Owns a device handle → non-copyable, movable.
    CudaRenderer(const CudaRenderer&)            = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;
    CudaRenderer(CudaRenderer&& other) noexcept;
    CudaRenderer& operator=(CudaRenderer&& other) noexcept;

    RenderOutput render(const RenderInput& in) const;

    const Eigen::MatrixX3i& triangles() const { return triangles_; }

private:
    int              H_;
    int              W_;
    Eigen::MatrixX3i triangles_;   // (M,3)
    void*            handle_;      // opaque CudaRasterState* (cuda_raster.cu)
};
