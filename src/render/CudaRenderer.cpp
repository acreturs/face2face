// host side of the cuda rasteriser
// it runs the same vertex stage as the cpu Renderer so the pixels, normals
// and colours match, then flattens everything into plain float arrays and
// hands them to the launchers in cuda_raster.cu through a C ABI
// on purpose there are no cuda headers here so g++ can compile this file and
// it just links against the nvcc built object
#include "CudaRenderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"

#include <utility>
#include <vector>

// these launchers live in cuda_raster.cu, built by nvcc
extern "C" {
void* cudaRasterCreate(int H, int W, const int* tris, int numTris);
void  cudaRasterRender(void* handle,
                       const float* vcam, const float* uv, const float* shaded,
                       int numVerts,
                       float* image, float* depth, unsigned char* mask,
                       int* triIdx, float* bary);
void  cudaRasterDestroy(void* handle);
}

CudaRenderer::CudaRenderer(int height, int width, Eigen::MatrixX3i triangles)
    : H_(height), W_(width), triangles_(std::move(triangles))
{
    // eigen is column major by default so flatten the triangle list into a
    // row major buffer of i0 i1 i2 per face that the kernel can index directly
    const int M = static_cast<int>(triangles_.rows());
    std::vector<int> tris(static_cast<size_t>(M) * 3);
    for (int f = 0; f < M; ++f) {
        tris[3 * f + 0] = triangles_(f, 0);
        tris[3 * f + 1] = triangles_(f, 1);
        tris[3 * f + 2] = triangles_(f, 2);
    }
    handle_ = cudaRasterCreate(H_, W_, tris.data(), M);
}

CudaRenderer::~CudaRenderer()
{
    if (handle_) cudaRasterDestroy(handle_);
}

CudaRenderer::CudaRenderer(CudaRenderer&& o) noexcept
    : H_(o.H_), W_(o.W_), triangles_(std::move(o.triangles_)), handle_(o.handle_)
{
    o.handle_ = nullptr;
}

CudaRenderer& CudaRenderer::operator=(CudaRenderer&& o) noexcept
{
    if (this != &o) {
        if (handle_) cudaRasterDestroy(handle_);
        H_ = o.H_; W_ = o.W_;
        triangles_ = std::move(o.triangles_);
        handle_ = o.handle_;
        o.handle_ = nullptr;
    }
    return *this;
}

RenderOutput CudaRenderer::render(const RenderInput& in) const
{
    const int N = static_cast<int>(in.shape.rows());

    // vertex stage on the cpu, same as Renderer::render
    const Eigen::MatrixX3f V_cam  = proj::toCameraFrame(in.shape, in.R, in.t);
    const proj::Pixels     uv     = proj::project(V_cam, in.K);
    const Eigen::MatrixX3f Nrm    = Renderer::computeNormals(in.shape, triangles_);
    const Eigen::MatrixX3f N_cam  = proj::normalsToCameraFrame(Nrm, in.R);
    const Eigen::MatrixX3f shaded = light::shadeVertices(in.albedo, N_cam, in.sh);

    // flatten into row major host arrays for the kernel
    std::vector<float> hVcam(static_cast<size_t>(N) * 3);
    std::vector<float> hUv  (static_cast<size_t>(N) * 2);
    std::vector<float> hShad(static_cast<size_t>(N) * 3);
    for (int i = 0; i < N; ++i) {
        hVcam[3 * i + 0] = V_cam(i, 0);
        hVcam[3 * i + 1] = V_cam(i, 1);
        hVcam[3 * i + 2] = V_cam(i, 2);
        hUv  [2 * i + 0] = uv(i, 0);
        hUv  [2 * i + 1] = uv(i, 1);
        hShad[3 * i + 0] = shaded(i, 0);
        hShad[3 * i + 1] = shaded(i, 1);
        hShad[3 * i + 2] = shaded(i, 2);
    }

    // cv::Mat storage is contiguous row major so the kernel can copy straight in
    RenderOutput out;
    out.image  = cv::Mat(H_, W_, CV_32FC3);
    out.depth  = cv::Mat(H_, W_, CV_32F);
    out.mask   = cv::Mat(H_, W_, CV_8U);
    out.triIdx = cv::Mat(H_, W_, CV_32S);
    out.bary   = cv::Mat(H_, W_, CV_32FC3);

    cudaRasterRender(handle_, hVcam.data(), hUv.data(), hShad.data(), N,
                     reinterpret_cast<float*>(out.image.data),
                     reinterpret_cast<float*>(out.depth.data),
                     out.mask.data,
                     reinterpret_cast<int*>(out.triIdx.data),
                     reinterpret_cast<float*>(out.bary.data));
    return out;
}
