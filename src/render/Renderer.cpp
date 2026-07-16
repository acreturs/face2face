// cpu renderer that turns a face mesh into an image
// projects the mesh, rasterises the triangles and fills in the shaded colours
#include "Renderer.h"
#include "ProjectionUtils.h"
#include "Lighting.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// triangles and image size stay fixed for the whole run
Renderer::Renderer(int height, int width, Eigen::MatrixX3i triangles)
    : H_(height), W_(width), triangles_(std::move(triangles)) {}

// runs the whole render, project then rasterise then shade
RenderOutput Renderer::render(const RenderInput& in) const
{
    RenderOutput out;
    out.image  = cv::Mat::zeros(H_, W_, CV_32FC3);
    out.depth  = cv::Mat(H_, W_, CV_32F, std::numeric_limits<float>::infinity());
    out.mask   = cv::Mat::zeros(H_, W_, CV_8U);
    out.triIdx = cv::Mat(H_, W_, CV_32S, cv::Scalar(-1));
    out.bary   = cv::Mat::zeros(H_, W_, CV_32FC3);

    // project the mesh into the camera frame
    const Eigen::MatrixX3f V_cam = proj::toCameraFrame(in.shape, in.R, in.t);
    const proj::Pixels     uv    = proj::project(V_cam, in.K);
    // normals go to the camera frame too since the shading needs them
    const Eigen::MatrixX3f N      = computeNormals(in.shape, triangles_);
    const Eigen::MatrixX3f N_cam  = proj::normalsToCameraFrame(N, in.R);

    // drop triangles that face away from the camera
    const std::vector<bool> frontFacing = backfaceMask(V_cam, triangles_);

    // rasterise the triangles into the buffers
    rasterize(uv, V_cam, frontFacing, out);

    // shade each vertex with the spherical harmonics light
    const Eigen::MatrixX3f shaded = light::shadeVertices(in.albedo, N_cam, in.sh);

    // blend the vertex colours across every covered pixel
    interpolateShading(shaded, out);
    return out;
}

// for each covered pixel mix the triangle's three vertex colours using the
// barycentric weights we stored, so colour = b0*C0 + b1*C1 + b2*C2
void Renderer::interpolateShading(const Eigen::MatrixX3f& shadedColors,
                                  RenderOutput&           out) const
{
    for (int y = 0; y < H_; ++y) {
        for (int x = 0; x < W_; ++x) {
            const int f = out.triIdx.at<int>(y, x);
            if (f < 0) continue;                      // nothing drawn here so skip

            const cv::Vec3f bary = out.bary.at<cv::Vec3f>(y, x);
            const int index0 = triangles_(f, 0);
            const int index1 = triangles_(f, 1);
            const int index2 = triangles_(f, 2);

            const Eigen::RowVector3f colour = bary[0] * shadedColors.row(index0) + bary[1] * shadedColors.row(index1) + bary[2] * shadedColors.row(index2);
            out.image.at<cv::Vec3f>(y, x) = cv::Vec3f(colour(0), colour(1), colour(2));
        }
    }
}


// z-buffer rasteriser
// for each front facing triangle we walk the pixels in its bounding box,
// find the barycentric weights and keep the nearest fragment per pixel
// we store the triangle index and the weights so a pixel colour stays a
// linear blend of 3 vertices, which is what makes it differentiable later
void Renderer::rasterize(const proj::Pixels&      uv,
                         const Eigen::MatrixX3f&  V_cam,
                         const std::vector<bool>& frontFacing,
                         RenderOutput&            out) const
{
    for (int f = 0; f < triangles_.rows(); ++f) {
        if (!frontFacing[f]) continue;

        const int vertex0 = triangles_(f, 0);
        const int vertex1 = triangles_(f, 1);
        const int vertex2 = triangles_(f, 2);

        // projected pixel positions
        const float x0 = uv(vertex0, 0), y0 = uv(vertex0, 1);
        const float x1 = uv(vertex1, 0), y1 = uv(vertex1, 1);
        const float x2 = uv(vertex2, 0), y2 = uv(vertex2, 1);

        // camera depth of each vertex for the perspective correction below
        const float z0 = V_cam(vertex0, 2);
        const float z1 = V_cam(vertex1, 2);
        const float z2 = V_cam(vertex2, 2);

        // skip if any vertex is behind the camera
        if (z0 <= 1e-6f || z1 <= 1e-6f || z2 <= 1e-6f) continue;

        // bounding box of the triangle clipped to the image
        int xmin = static_cast<int>(std::floor(std::min({x0, x1, x2})));
        int xmax = static_cast<int>(std::ceil (std::max({x0, x1, x2})));
        int ymin = static_cast<int>(std::floor(std::min({y0, y1, y2})));
        int ymax = static_cast<int>(std::ceil (std::max({y0, y1, y2})));
        xmin = std::max(0, xmin);  
        xmax = std::min(W_ - 1, xmax);
        ymin = std::max(0, ymin);  
        ymax = std::min(H_ - 1, ymax);
        if (xmin > xmax || ymin > ymax) continue;     // fully off screen

        // edge function denominator, twice the signed triangle area in screen space
        // same value as the culling n_z, just reused here for the barycentrics
        const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (std::abs(denom) < 1e-12f) continue;        // degenerate triangle
        const float invDenom = 1.0f / denom;

        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                // sample at the pixel centre
                const float px = x + 0.5f;
                const float py = y + 0.5f;

                // screen space barycentric weights
                const float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * invDenom;
                const float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * invDenom;
                const float w2 = 1.0f - w0 - w1;

                // inside the triangle when all weights are non negative
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

                // perspective correct depth
                // screen space barycentrics are wrong for interpolating on a
                // perspective triangle, but 1/z is linear in screen space so
                // we work through that
                const float inv_z = w0 / z0 + w1 / z1 + w2 / z2;
                const float z_pixel = 1.0f / inv_z;

                // perspective correct weights b_k = (w_k/z_k)/inv_z
                // and since z_pixel = 1/inv_z that is (w_k/z_k)*z_pixel
                const float b0 = w0 / z0 * z_pixel;
                const float b1 = w1 / z1 * z_pixel;
                const float b2 = w2 / z2 * z_pixel;

                // keep the nearest fragment per pixel
                if (z_pixel < out.depth.at<float>(y, x)) {
                    out.depth.at<float>(y, x) = z_pixel;
                    out.triIdx.at<int>(y, x) = f;
                    out.bary.at<cv::Vec3f>(y, x) = cv::Vec3f(b0, b1, b2);
                    out.mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }
}

// back face culling in the camera frame
// this is a camera frame test not a screen space winding test, because the
// winding test breaks under the BFM_TO_CAM y flip
std::vector<bool> Renderer::backfaceMask(const Eigen::MatrixX3f& V_cam,
                                         const Eigen::MatrixX3i& triangles)
{
    std::vector<bool> frontFacing;
    frontFacing.reserve(triangles.rows());

    for (int f = 0; f < triangles.rows(); ++f) {
        const Eigen::Vector3f v0 = V_cam.row(triangles(f, 0));
        const Eigen::Vector3f v1 = V_cam.row(triangles(f, 1));
        const Eigen::Vector3f v2 = V_cam.row(triangles(f, 2));

        // a triangle faces the camera when its normal points against the view
        // ray to it, so n·v0 < 0 with the camera at the origin
        // the plain n_z test is only the orthographic version and gets grazing
        // triangles near the silhouette wrong, which is where the matching relies
        const Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
        frontFacing.push_back(n.dot(v0) < 0.0f);
    }

    return frontFacing;
}

// area weighted unit normals per vertex
// the face normal (v1-v0) x (v2-v0) is left un normalised so its length is
// twice the triangle area, which makes bigger faces count more
// we add it onto the 3 vertices then normalise each vertex at the end
Eigen::MatrixX3f Renderer::computeNormals(const Eigen::MatrixX3f& shape,
                                          const Eigen::MatrixX3i& triangles)
{
    Eigen::MatrixX3f normals = Eigen::MatrixX3f::Zero(shape.rows(), 3);

    for (int f = 0; f < triangles.rows(); ++f) {
        const int i0 = triangles(f, 0);
        const int i1 = triangles(f, 1);
        const int i2 = triangles(f, 2);

        const Eigen::Vector3f v0 = shape.row(i0);
        const Eigen::Vector3f v1 = shape.row(i1);
        const Eigen::Vector3f v2 = shape.row(i2);

        const Eigen::Vector3f faceNormal = (v1 - v0).cross(v2 - v0);

        normals.row(i0) += faceNormal;
        normals.row(i1) += faceNormal;
        normals.row(i2) += faceNormal;
    }

    for (int i = 0; i < normals.rows(); ++i) {
        const float n = normals.row(i).norm();
        if (n > 1e-12f) normals.row(i) /= n; // avoid dividing by zero on degenerate vertices
    }
    return normals;
}
