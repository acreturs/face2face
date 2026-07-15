#pragma once
#include <Eigen/Dense>
#include "Renderer.h"   // shares RenderInput and RenderOutput

// the GPU version of Renderer with the exact same interface so it drops
// straight into the spots that use a renderer (see runLiveImpl and
// runTransferLive in main). only the heavy per-pixel rasterising runs on the
// GPU, the cheap per-vertex math still runs on the CPU with the same helpers
// so the result matches the CPU renderer.
// no CUDA headers leak out of here, the device state is just a void* so plain
// g++ files can include this. only compiled with make USE_CUDA=1
class CudaRenderer {
public:
    // size and triangles are fixed for the run so we upload them once
    CudaRenderer(int height, int width, Eigen::MatrixX3i triangles);
    ~CudaRenderer();

    // owns GPU memory so no copying, moving is fine
    CudaRenderer(const CudaRenderer&)            = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;
    CudaRenderer(CudaRenderer&& other) noexcept;
    CudaRenderer& operator=(CudaRenderer&& other) noexcept;

    RenderOutput render(const RenderInput& in) const;

    const Eigen::MatrixX3i& triangles() const { return triangles_; }

private:
    int              H_;
    int              W_;
    Eigen::MatrixX3i triangles_;
    void*            handle_;   // opaque GPU state, lives in cuda_raster.cu
};
