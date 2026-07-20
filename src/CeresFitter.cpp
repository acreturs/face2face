#include "CeresFitter.h"
#include "ProjectionUtils.h"
#include "Renderer.h"
#include "Lighting.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>

#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <fstream>
#include <random>
#include <iostream>
#include <stdexcept>

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <utility>

// GPU photometric inner solve — off unless --photo-gpu (and a USE_CUDA build).
bool        CeresFitter::usePhotometricGpu = false;
bool        CeresFitter::photoGpuAnalytic  = false;
int         CeresFitter::photoSamples      = 0;
std::string CeresFitter::photoCsvPath;

#ifdef USE_CUDA
// Launchers implemented in src/render/cuda_photometric.cu (built by nvcc).
extern "C" {
void*  cudaPhotoPrepare(const float* image, int H, int W,
                        float fx, float fy, float cx, float cy,
                        int P, const double* basePoint, const double* bBasis,
                        const double* target, int K, int optimizeShape);
void   cudaPhotoNormalEq(void* h, const double* aa, const double* t, const double* shape,
                         int optimizePose, int optimizeShape, int analytic,
                         double sqrtWeight, double huberDelta,
                         double* JtJ, double* Jtr, double* cost);
double cudaPhotoCost(void* h, const double* aa, const double* t, const double* shape,
                     int optimizePose, int optimizeShape,
                     double sqrtWeight, double huberDelta);
void   cudaPhotoDestroy(void* h);
}
#endif

namespace {

struct LandmarkReprojectionResidual {
    LandmarkReprojectionResidual(
        const Eigen::Vector3f& modelPoint,
        const Eigen::Vector2d& imagePoint,
        const Eigen::Matrix3f& intrinsics
    )
        : observedU_(imagePoint.x()),
          observedV_(imagePoint.y()),
          fx_(intrinsics(0, 0)),
          fy_(intrinsics(1, 1)),
          cx_(intrinsics(0, 2)),
          cy_(intrinsics(1, 2))
    {
        // Match ProjectionUtils:
        // cameraPoint = R * BFM_TO_CAM * modelPoint + translation
        const Eigen::Vector3d alignedPoint = proj::BFM_TO_CAM.cast<double>() * modelPoint.cast<double>();

        modelX_ = alignedPoint.x();
        modelY_ = alignedPoint.y();
        modelZ_ = alignedPoint.z();
    }

    template <typename T>
    bool operator()(
        const T* const angleAxis,
        const T* const translation,
        T* residuals
    ) const {
        const T modelPoint[3] = {
            T(modelX_),
            T(modelY_),
            T(modelZ_)
        };

        T rotatedPoint[3];

        ceres::AngleAxisRotatePoint(
            angleAxis,
            modelPoint,
            rotatedPoint
        );

        const T cameraX = rotatedPoint[0] + translation[0];
        const T cameraY = rotatedPoint[1] + translation[1];
        const T cameraZ = rotatedPoint[2] + translation[2];

        const T projectedU =
            T(fx_) * cameraX / cameraZ + T(cx_);

        const T projectedV =
            T(fy_) * cameraY / cameraZ + T(cy_);

        residuals[0] = projectedU - T(observedU_);
        residuals[1] = projectedV - T(observedV_);

        return true;
    }

    

private:
    double modelX_;
    double modelY_;
    double modelZ_;

    double observedU_;
    double observedV_;

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};

struct LandmarkShapeReprojectionResidual{
    LandmarkShapeReprojectionResidual(
        const Eigen::Vector3f& meanPoint,
        int vertexIndex,
        const Eigen::MatrixXf& shapeBasis,
        const Eigen::VectorXf& shapeSigma,
        const Eigen::Vector2d& imagePoint,
        const Eigen::Matrix3f& intrinsics
    )
        : observedU_(imagePoint.x()),
          observedV_(imagePoint.y()),
          fx_(intrinsics(0, 0)),
          fy_(intrinsics(1, 1)),
          cx_(intrinsics(0, 2)),
          cy_(intrinsics(1, 2))
    {
        const Eigen::Vector3d alignedMean =
            proj::BFM_TO_CAM.cast<double>() *
            meanPoint.cast<double>();

        meanX_ = alignedMean.x();
        meanY_ = alignedMean.y();
        meanZ_ = alignedMean.z();

        for (int coefficientIndex = 0;
             coefficientIndex < kShapeCoefficientCount;
             ++coefficientIndex) {

            Eigen::Vector3d displacement;

            displacement.x() =
                shapeBasis(3 * vertexIndex + 0, coefficientIndex) *
                shapeSigma(coefficientIndex);

            displacement.y() =
                shapeBasis(3 * vertexIndex + 1, coefficientIndex) *
                shapeSigma(coefficientIndex);

            displacement.z() =
                shapeBasis(3 * vertexIndex + 2, coefficientIndex) *
                shapeSigma(coefficientIndex);

            const Eigen::Vector3d alignedDisplacement =
                proj::BFM_TO_CAM.cast<double>() * displacement;

            basisX_[coefficientIndex] = alignedDisplacement.x();
            basisY_[coefficientIndex] = alignedDisplacement.y();
            basisZ_[coefficientIndex] = alignedDisplacement.z();
        }
    }

    template <typename T>
    bool operator()(
        const T* const angleAxis,
        const T* const translation,
        const T* const shapeCoefficients,
        T* residuals
    ) const {
        T modelPoint[3] = {
            T(meanX_),
            T(meanY_),
            T(meanZ_)
        };

        for (int coefficientIndex = 0;
             coefficientIndex < kShapeCoefficientCount;
             ++coefficientIndex) {

            modelPoint[0] +=
                T(basisX_[coefficientIndex]) *
                shapeCoefficients[coefficientIndex];

            modelPoint[1] +=
                T(basisY_[coefficientIndex]) *
                shapeCoefficients[coefficientIndex];

            modelPoint[2] +=
                T(basisZ_[coefficientIndex]) *
                shapeCoefficients[coefficientIndex];
        }

        T rotatedPoint[3];

        ceres::AngleAxisRotatePoint(
            angleAxis,
            modelPoint,
            rotatedPoint
        );

        const T cameraX =
            rotatedPoint[0] + translation[0];

        const T cameraY =
            rotatedPoint[1] + translation[1];

        const T cameraZ =
            rotatedPoint[2] + translation[2];

        const T projectedU =
            T(fx_) * cameraX / cameraZ + T(cx_);

        const T projectedV =
            T(fy_) * cameraY / cameraZ + T(cy_);

        residuals[0] =
            projectedU - T(observedU_);

        residuals[1] =
            projectedV - T(observedV_);

        return true;
    }

private:
    double meanX_;
    double meanY_;
    double meanZ_;

    std::array<double, kShapeCoefficientCount> basisX_;
    std::array<double, kShapeCoefficientCount> basisY_;
    std::array<double, kShapeCoefficientCount> basisZ_;

    double observedU_;
    double observedV_;

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};

struct ShapeRegularizationResidual {
    explicit ShapeRegularizationResidual(
        double regularizationWeight
    )
        : sqrtWeight_(std::sqrt(regularizationWeight)) {}

    template <typename T>
    bool operator()(
        const T* const shapeCoefficients,
        T* residuals
    ) const {
        for (int coefficientIndex = 0;
             coefficientIndex < kShapeCoefficientCount;
             ++coefficientIndex) {

            residuals[coefficientIndex] =
                T(sqrtWeight_) *
                shapeCoefficients[coefficientIndex];
        }

        return true;
    }

private:
    double sqrtWeight_;
};

// Dense geometry residual (Eq. 2): point-to-point plus point-to-plane. The
// residual is the full 3D difference (R·v + t) − targetPoint (3 components) plus
// its component along the surface normal (1 component). Already in the camera
// frame, so no projection; targetNormal is constant, like targetPoint.
struct DepthPointResidual {
    DepthPointResidual(
        const Eigen::Vector3f& meanPoint,
        int vertexIndex,
        const Eigen::MatrixXf& shapeBasis,
        const Eigen::VectorXf& shapeSigma,
        const Eigen::Vector3d& targetPoint,
        const Eigen::Vector3d& targetNormal,
        double pointToPlaneWeight,
        // Current expression displacement of this vertex (camera-aligned mm),
        // held constant within the solve: otherwise the residual models the
        // neutral vertex while the correspondence was found against the expressed
        // shape, biasing pose/identity. Refreshed each ICP iteration.
        const Eigen::Vector3d& exprOffsetAligned = Eigen::Vector3d::Zero()
    )
        : targetX_(targetPoint.x()),
          targetY_(targetPoint.y()),
          targetZ_(targetPoint.z()),
          normalX_(targetNormal.x()),
          normalY_(targetNormal.y()),
          normalZ_(targetNormal.z()),
          sqrtPlaneWeight_(std::sqrt(pointToPlaneWeight))
    {
        const Eigen::Vector3d alignedMean =
            proj::BFM_TO_CAM.cast<double>() * meanPoint.cast<double>() +
            exprOffsetAligned;

        meanX_ = alignedMean.x();
        meanY_ = alignedMean.y();
        meanZ_ = alignedMean.z();

        for (int coefficientIndex = 0;
             coefficientIndex < kShapeCoefficientCount;
             ++coefficientIndex) {

            Eigen::Vector3d displacement;
            displacement.x() =
                shapeBasis(3 * vertexIndex + 0, coefficientIndex) *
                shapeSigma(coefficientIndex);
            displacement.y() =
                shapeBasis(3 * vertexIndex + 1, coefficientIndex) *
                shapeSigma(coefficientIndex);
            displacement.z() =
                shapeBasis(3 * vertexIndex + 2, coefficientIndex) *
                shapeSigma(coefficientIndex);

            const Eigen::Vector3d alignedDisplacement =
                proj::BFM_TO_CAM.cast<double>() * displacement;

            basisX_[coefficientIndex] = alignedDisplacement.x();
            basisY_[coefficientIndex] = alignedDisplacement.y();
            basisZ_[coefficientIndex] = alignedDisplacement.z();
        }
    }

    template <typename T>
    bool operator()(
        const T* const angleAxis,
        const T* const translation,
        const T* const shapeCoefficients,
        T* residuals
    ) const {
        T modelPoint[3] = { T(meanX_), T(meanY_), T(meanZ_) };

        for (int coefficientIndex = 0;
             coefficientIndex < kShapeCoefficientCount;
             ++coefficientIndex) {
            modelPoint[0] += T(basisX_[coefficientIndex]) * shapeCoefficients[coefficientIndex];
            modelPoint[1] += T(basisY_[coefficientIndex]) * shapeCoefficients[coefficientIndex];
            modelPoint[2] += T(basisZ_[coefficientIndex]) * shapeCoefficients[coefficientIndex];
        }

        T rotatedPoint[3];
        ceres::AngleAxisRotatePoint(angleAxis, modelPoint, rotatedPoint);

        // point-to-point: full 3D difference model point − depth point.
        const T dx = (rotatedPoint[0] + translation[0]) - T(targetX_);
        const T dy = (rotatedPoint[1] + translation[1]) - T(targetY_);
        const T dz = (rotatedPoint[2] + translation[2]) - T(targetZ_);
        residuals[0] = dx;
        residuals[1] = dy;
        residuals[2] = dz;
        // point-to-plane: component of the difference along the surface normal
        // (lets the mesh slide along the tangent → more stable/faster).
        residuals[3] = T(sqrtPlaneWeight_) *
                       (T(normalX_) * dx + T(normalY_) * dy + T(normalZ_) * dz);
        return true;
    }

private:
    double meanX_, meanY_, meanZ_;
    std::array<double, kShapeCoefficientCount> basisX_;
    std::array<double, kShapeCoefficientCount> basisY_;
    std::array<double, kShapeCoefficientCount> basisZ_;
    double targetX_, targetY_, targetZ_;
    double normalX_, normalY_, normalZ_;
    double sqrtPlaneWeight_;
};

// A 3-channel (RGB), row-major, differentiable image sampler. Ceres' bicubic
// interpolator gives us the observed colour AND its derivative w.r.t. the sample
// position, so the photometric residual is differentiable w.r.t. pose/shape.
using PhotoGrid   = ceres::Grid2D<double, 3>;
using PhotoInterp = ceres::BiCubicInterpolator<PhotoGrid>;

// Per-pixel photometric residual, driven by the renderer's G-buffer. For one
// covered pixel:
//   S = b0·v0 + b1·v1 + b2·v2                       (barycentric surface point)
//   residual = w · ( renderedColour(pixel) − inputImage(project(R·S + t)) )
// The barycentric weights and the target colour come straight from render();
// correspondence (which triangle covers the pixel) is fixed per outer iteration,
// like ICP. Since Σb = 1 and (R,t) is affine, the three vertices pre-blend into
// one effective point, so the functor is a single projected sample whose colour
// moves with pose + shape — that is the photometric gradient.
struct PhotometricPixelResidual {
    PhotometricPixelResidual(
        const Eigen::Vector3d&                                      blendedMeanAligned,
        const std::array<Eigen::Vector3d, kShapeCoefficientCount>&  blendedBasisAligned,
        const Eigen::Vector3d&  targetColor,       
        const Eigen::Matrix3f&  intrinsics,
        const PhotoInterp&      image,
        double                  sqrtWeight
    )
        : image_(image),
          sqrtWeight_(sqrtWeight),
          tgtR_(targetColor.x()),
          tgtG_(targetColor.y()),
          tgtB_(targetColor.z()),
          fx_(intrinsics(0, 0)),
          fy_(intrinsics(1, 1)),
          cx_(intrinsics(0, 2)),
          cy_(intrinsics(1, 2)),
          meanX_(blendedMeanAligned.x()),
          meanY_(blendedMeanAligned.y()),
          meanZ_(blendedMeanAligned.z())
    {
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            basisX_[k] = blendedBasisAligned[k].x();
            basisY_[k] = blendedBasisAligned[k].y();
            basisZ_[k] = blendedBasisAligned[k].z();
        }
    }

    template <typename T>
    bool operator()(
        const T* const angleAxis,
        const T* const translation,
        const T* const shapeCoefficients,
        T*             residuals
    ) const {
        T surfacePoint[3] = { T(meanX_), T(meanY_), T(meanZ_) };
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            surfacePoint[0] += T(basisX_[k]) * shapeCoefficients[k];
            surfacePoint[1] += T(basisY_[k]) * shapeCoefficients[k];
            surfacePoint[2] += T(basisZ_[k]) * shapeCoefficients[k];
        }

        T rotated[3];
        ceres::AngleAxisRotatePoint(angleAxis, surfacePoint, rotated);
        const T cameraX = rotated[0] + translation[0];
        const T cameraY = rotated[1] + translation[1];
        const T cameraZ = rotated[2] + translation[2];

        const T u = T(fx_) * cameraX / cameraZ + T(cx_);
        const T v = T(fy_) * cameraY / cameraZ + T(cy_);

        // Differentiable input-image lookup. Grid is (row=v, col=u); out-of-range
        // samples are clamped to the border by Grid2D
        T observed[3];
        image_.Evaluate(v, u, observed);

        residuals[0] = T(sqrtWeight_) * (T(tgtR_) - observed[0]);
        residuals[1] = T(sqrtWeight_) * (T(tgtG_) - observed[1]);
        residuals[2] = T(sqrtWeight_) * (T(tgtB_) - observed[2]);
        return true;
    }

private:
    const PhotoInterp& image_;
    double sqrtWeight_;
    double tgtR_, tgtG_, tgtB_;
    double fx_, fy_, cx_, cy_;
    double meanX_, meanY_, meanZ_;
    std::array<double, kShapeCoefficientCount> basisX_;
    std::array<double, kShapeCoefficientCount> basisY_;
    std::array<double, kShapeCoefficientCount> basisZ_;
};

// Landmark reprojection with identity + expression coefficients. Parameter
// blocks: angleAxis(3), translation(3), identity, expr:
//   modelPoint = mean + idBasis·alpha + exprBasis·delta   (then pose + project)
// This is what gives the contour fit an expression gradient. The focal-aware
// variants make fx=fy a parameter block so an uncalibrated webcam's focal can be
// solved (identical to fixed intrinsics when the block is held constant).
// WithExpr toggles the expression basis; WithId=false expects the identity
// displacement pre-baked into meanPoint and drops the block — during tracking
// (identity frozen) that avoids ~3× the autodiff cost of a constant 180-coeff block.
template <bool WithExpr, bool WithId = true>
struct LandmarkFocalReprojectionResidual {
    LandmarkFocalReprojectionResidual(
        const Eigen::Vector3f& meanPoint, int vertexIndex,
        const Eigen::MatrixXf& shapeBasis, const Eigen::VectorXf& shapeSigma,
        const Eigen::MatrixXf& exprBasis,  const Eigen::VectorXf& exprSigma,
        const Eigen::Vector2d& imagePoint, const Eigen::Matrix3f& intrinsics)
        : observedU_(imagePoint.x()), observedV_(imagePoint.y()),
          cx_(intrinsics(0, 2)), cy_(intrinsics(1, 2))
    {
        const Eigen::Matrix3d M = proj::BFM_TO_CAM.cast<double>();
        const Eigen::Vector3d m = M * meanPoint.cast<double>();
        meanX_ = m.x(); meanY_ = m.y(); meanZ_ = m.z();
        if constexpr (WithId)
            for (int k = 0; k < kShapeCoefficientCount; ++k) {
                const Eigen::Vector3d a = M * Eigen::Vector3d(
                    shapeBasis(3 * vertexIndex + 0, k) * shapeSigma(k),
                    shapeBasis(3 * vertexIndex + 1, k) * shapeSigma(k),
                    shapeBasis(3 * vertexIndex + 2, k) * shapeSigma(k));
                idX_[k] = a.x(); idY_[k] = a.y(); idZ_[k] = a.z();
            }
        if constexpr (WithExpr)
            for (int j = 0; j < kExpressionCoefficientCount; ++j) {
                const Eigen::Vector3d a = M * Eigen::Vector3d(
                    exprBasis(3 * vertexIndex + 0, j) * exprSigma(j),
                    exprBasis(3 * vertexIndex + 1, j) * exprSigma(j),
                    exprBasis(3 * vertexIndex + 2, j) * exprSigma(j));
                exX_[j] = a.x(); exY_[j] = a.y(); exZ_[j] = a.z();
            }
    }

    // Shared core: build the model point, pose it, project with focal[0].
    template <typename T>
    bool project(const T* angleAxis, const T* translation, const T* idCoeff,
                 const T* exprCoeff, const T* focal, T* residuals) const {
        T p[3] = { T(meanX_), T(meanY_), T(meanZ_) };
        if constexpr (WithId)
            for (int k = 0; k < kShapeCoefficientCount; ++k) {
                p[0] += T(idX_[k]) * idCoeff[k];
                p[1] += T(idY_[k]) * idCoeff[k];
                p[2] += T(idZ_[k]) * idCoeff[k];
            }
        if constexpr (WithExpr)
            for (int j = 0; j < kExpressionCoefficientCount; ++j) {
                p[0] += T(exX_[j]) * exprCoeff[j];
                p[1] += T(exY_[j]) * exprCoeff[j];
                p[2] += T(exZ_[j]) * exprCoeff[j];
            }
        T r[3];
        ceres::AngleAxisRotatePoint(angleAxis, p, r);
        const T X = r[0] + translation[0];
        const T Y = r[1] + translation[1];
        const T Z = r[2] + translation[2];
        residuals[0] = focal[0] * X / Z + T(cx_) - T(observedU_);
        residuals[1] = focal[0] * Y / Z + T(cy_) - T(observedV_);
        return true;
    }

    // 5 blocks — only <WithExpr=true, WithId=true>: aa, t, id, expr, focal.
    // Ceres calls the overload matching the declared block count, so the
    // unused arities are never instantiated.
    template <typename T>
    bool operator()(const T* const angleAxis, const T* const translation,
                    const T* const idCoeff, const T* const exprCoeff,
                    const T* const focal, T* residuals) const {
        return project(angleAxis, translation, idCoeff, exprCoeff, focal, residuals);
    }
    template <typename T>
    bool operator()(const T* const angleAxis, const T* const translation,
                    const T* const coeff, const T* const focal,
                    T* residuals) const {
        if constexpr (WithId)
            return project(angleAxis, translation, coeff,
                           static_cast<const T*>(nullptr), focal, residuals);
        else
            return project(angleAxis, translation,
                           static_cast<const T*>(nullptr), coeff, focal, residuals);
    }

private:
    // Conditionally-sized storage: a dummy 1-element array when the basis is
    // compiled out, so the frozen-identity functor doesn't carry 3×180 doubles.
    double meanX_, meanY_, meanZ_;
    std::array<double, WithId ? kShapeCoefficientCount : 1>      idX_, idY_, idZ_;
    std::array<double, WithExpr ? kExpressionCoefficientCount : 1> exX_, exY_, exZ_;
    double observedU_, observedV_, cx_, cy_;
};

// Weak prior anchoring the focal estimate at its initial guess. The focal↔
// distance ambiguity is nearly flat, so an unanchored focal rails at its bound;
// this keeps it in the guess's basin while the perspective signal (face depth
// variation) applies the real correction.
struct FocalPriorResidual {
    FocalPriorResidual(double f0, double weight) : f0_(f0), w_(weight) {}
    template <typename T>
    bool operator()(const T* const focal, T* residual) const {
        residual[0] = T(w_) * (focal[0] - T(f0_)) / T(f0_);
        return true;
    }
    double f0_, w_;
};

// Generic L2 prior (√w · coeff) on a compile-time-sized coefficient block — used
// for the expression prior, mirroring ShapeRegularizationResidual for identity.
template <int Count>
struct CoeffPriorResidual {
    explicit CoeffPriorResidual(double weight) : sqrtWeight_(std::sqrt(weight)) {}
    template <typename T>
    bool operator()(const T* const coeff, T* residuals) const {
        for (int k = 0; k < Count; ++k) residuals[k] = T(sqrtWeight_) * coeff[k];
        return true;
    }
    double sqrtWeight_;
};

// L2 prior anchored at a target vector: √w·(coeff − target). Used as the temporal
// expression prior during tracking — a zero-anchored prior strong enough to damp
// jitter also pulls a held expression shut every frame, whereas this one only
// resists change, so holding an articulation costs nothing.
template <int Count>
struct CoeffAnchorResidual {
    CoeffAnchorResidual(double weight, const Eigen::VectorXd& target)
        : sqrtWeight_(std::sqrt(weight))
    {
        for (int k = 0; k < Count; ++k)
            target_[k] = k < target.size() ? target(k) : 0.0;
    }
    template <typename T>
    bool operator()(const T* const coeff, T* residuals) const {
        for (int k = 0; k < Count; ++k)
            residuals[k] = T(sqrtWeight_) * (coeff[k] - T(target_[k]));
        return true;
    }
    double sqrtWeight_;
    std::array<double, Count> target_;
};

} // namespace

// Estimate per-point cloud normals (defined below) — forward decl so the
// contour+depth fit can call it.
static std::vector<Eigen::Vector3d> estimateCloudNormals(
    const std::vector<Eigen::Vector3d>& cloud, int k);

Eigen::Matrix3f PoseParameters::rotationMatrix() const
{
    const double angle = angleAxis.norm();

    if (angle < 1e-12) {
        return Eigen::Matrix3f::Identity();
    }

    const Eigen::Vector3d axis = angleAxis / angle;

    return Eigen::AngleAxisd(angle, axis)
        .toRotationMatrix()
        .cast<float>();
}

std::vector<LandmarkObservation> loadLandmarkObservations(
    const std::string& path
)
{
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Could not open landmark file: " + path
        );
    }

    std::vector<LandmarkObservation> observations;

    int vertexIndex;
    double u;
    double v;

    while (input >> vertexIndex >> u >> v) {
        observations.push_back({
            vertexIndex,
            Eigen::Vector2d(u, v)
        });
    }

    if (observations.empty()) {
        throw std::runtime_error(
            "No landmark observations found in: " + path
        );
    }

    return observations;
}

PoseParameters CeresFitter::fitPose(
    const Eigen::MatrixX3f& shape,
    const std::vector<LandmarkObservation>& observations,
    const Eigen::Matrix3f& intrinsics,
    const PoseParameters& initialPose,
    double zMin,
    double zMax
)
{
    double angleAxis[3] = {
        initialPose.angleAxis.x(),
        initialPose.angleAxis.y(),
        initialPose.angleAxis.z()
    };

    double translation[3] = {
        initialPose.translation.x(),
        initialPose.translation.y(),
        initialPose.translation.z()
    };

    ceres::Problem problem;

    for (const LandmarkObservation& observation : observations) {
        if (
            observation.vertexIndex < 0 ||
            observation.vertexIndex >= shape.rows()
        ) {
            throw std::runtime_error(
                "Invalid BFM vertex index: " +
                std::to_string(observation.vertexIndex)
            );
        }

        const Eigen::Vector3f modelPoint =
            shape.row(observation.vertexIndex).transpose();

        ceres::CostFunction* costFunction =
            new ceres::AutoDiffCostFunction<
                LandmarkReprojectionResidual,
                2,
                3,
                3
            >(
                new LandmarkReprojectionResidual(
                    modelPoint,
                    observation.imagePoint,
                    intrinsics
                )
            );

        problem.AddResidualBlock(
            costFunction,
            new ceres::HuberLoss(10.0),   // px — robust to one bad detection
            angleAxis,
            translation
        );
    }

    // Keep the face in front of the camera (depth range depends on the dataset).
    problem.SetParameterLowerBound(translation, 2, zMin);
    problem.SetParameterUpperBound(translation, 2, zMax);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type =
        ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;

    std::cout << "\nStarting landmark pose fitting with "
              << observations.size()
              << " observations\n";

    ceres::Solve(
        options,
        &problem,
        &summary
    );

    std::cout << summary.BriefReport() << '\n';

    PoseParameters result;

    result.angleAxis = Eigen::Vector3d(
        angleAxis[0],
        angleAxis[1],
        angleAxis[2]
    );

    result.translation = Eigen::Vector3d(
        translation[0],
        translation[1],
        translation[2]
    );

    std::cout << "Optimized angle-axis: "
              << result.angleAxis.transpose() << '\n';

    std::cout << "Optimized translation: "
              << result.translation.transpose() << '\n';

    return result;
}

FitParameters CeresFitter::fitPoseAndShape(
    const Eigen::MatrixX3f& meanShape,
    const Eigen::MatrixXf& shapeBasis,
    const Eigen::VectorXf& shapeSigma,
    const std::vector<LandmarkObservation>& observations,
    const Eigen::Matrix3f& intrinsics,
    const PoseParameters& initialPose,
    double regularizationWeight,
    double zMin,
    double zMax
)
{
    if (shapeBasis.cols() < kShapeCoefficientCount) {
        throw std::runtime_error(
            "The BFM shape basis does not contain enough coefficients"
        );
    }

    if (shapeSigma.size() < kShapeCoefficientCount) {
        throw std::runtime_error(
            "The BFM shape sigma vector does not contain enough coefficients"
        );
    }

    double angleAxis[3] = {
        initialPose.angleAxis.x(),
        initialPose.angleAxis.y(),
        initialPose.angleAxis.z()
    };

    double translation[3] = {
        initialPose.translation.x(),
        initialPose.translation.y(),
        initialPose.translation.z()
    };

    double shapeCoefficients[kShapeCoefficientCount] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    };

    ceres::Problem problem;

    for (const LandmarkObservation& observation : observations) {
        if (
            observation.vertexIndex < 0 ||
            observation.vertexIndex >= meanShape.rows()
        ) {
            throw std::runtime_error(
                "Invalid BFM vertex index: " +
                std::to_string(observation.vertexIndex)
            );
        }

        const Eigen::Vector3f meanPoint =
            meanShape.row(observation.vertexIndex).transpose();

        ceres::CostFunction* costFunction =
            new ceres::AutoDiffCostFunction<
                LandmarkShapeReprojectionResidual,
                2,
                3,
                3,
                kShapeCoefficientCount
            >(
                new LandmarkShapeReprojectionResidual(
                    meanPoint,
                    observation.vertexIndex,
                    shapeBasis,
                    shapeSigma,
                    observation.imagePoint,
                    intrinsics
                )
            );

        problem.AddResidualBlock(
            costFunction,
            nullptr,
            angleAxis,
            translation,
            shapeCoefficients
        );
    }

    ceres::CostFunction* regularizationCost =
        new ceres::AutoDiffCostFunction<
            ShapeRegularizationResidual,
            kShapeCoefficientCount,
            kShapeCoefficientCount
        >(
            new ShapeRegularizationResidual(
                regularizationWeight
            )
        );

    problem.AddResidualBlock(
        regularizationCost,
        nullptr,
        shapeCoefficients
    );

    problem.SetParameterLowerBound(translation, 2, zMin);
    problem.SetParameterUpperBound(translation, 2, zMax);

    for (int axis = 0; axis < 3; ++axis) {
        problem.SetParameterLowerBound(
            angleAxis,
            axis,
            -0.7
        );

        problem.SetParameterUpperBound(
            angleAxis,
            axis,
            0.7
        );
    }

    for (int coefficientIndex = 0;
         coefficientIndex < kShapeCoefficientCount;
         ++coefficientIndex) {

        problem.SetParameterLowerBound(
            shapeCoefficients,
            coefficientIndex,
            -3.0
        );

        problem.SetParameterUpperBound(
            shapeCoefficients,
            coefficientIndex,
            3.0
        );
    }

    ceres::Solver::Options options;

    options.linear_solver_type =
        ceres::DENSE_QR;

    options.trust_region_strategy_type =
        ceres::LEVENBERG_MARQUARDT;

    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;

    std::cout
        << "\nStarting landmark pose and shape fitting with "
        << observations.size()
        << " observations and "
        << kShapeCoefficientCount
        << " shape coefficients\n";

    ceres::Solve(
        options,
        &problem,
        &summary
    );

    std::cout << summary.BriefReport() << '\n';

    FitParameters result;

    result.pose.angleAxis = Eigen::Vector3d(
        angleAxis[0],
        angleAxis[1],
        angleAxis[2]
    );

    result.pose.translation = Eigen::Vector3d(
        translation[0],
        translation[1],
        translation[2]
    );

    result.shapeCoefficients.resize(
        kShapeCoefficientCount
    );

    for (int coefficientIndex = 0;
         coefficientIndex < kShapeCoefficientCount;
         ++coefficientIndex) {

        result.shapeCoefficients(coefficientIndex) =
            shapeCoefficients[coefficientIndex];
    }

    std::cout << "Optimized angle-axis: "
              << result.pose.angleAxis.transpose()
              << '\n';

    std::cout << "Optimized translation: "
              << result.pose.translation.transpose()
              << '\n';

    std::cout << "Optimized shape coefficients: "
              << result.shapeCoefficients.transpose()
              << '\n';

    return result;
}

FitParameters CeresFitter::fitPoseAndShapeContour(
    const Eigen::MatrixX3f&                  meanShape,
    const Eigen::MatrixXf&                   shapeBasis,
    const Eigen::VectorXf&                   shapeSigma,
    const Eigen::MatrixXf&                   exprBasis,
    const Eigen::VectorXf&                   exprSigma,
    const Eigen::MatrixX3i&                  triangles,
    const std::vector<LandmarkObservation>&  observations,
    const Eigen::Matrix3f&                   intrinsics,
    const PoseParameters&                    initialPose,
    double                                   regularizationWeight,
    double                                   exprRegWeight,
    double                                   zMin,
    double                                   zMax,
    int                                      numOuterIterations,
    const std::vector<Eigen::Vector3d>*      targetCloud,
    double                                   depthPointToPlaneWeight,
    double                                   depthWeight,
    int                                      depthVertexStride,
    const Eigen::VectorXd&                   initialIdentity,
    const Eigen::VectorXd&                   initialExpr,
    bool                                     optimizeIdentity,
    bool                                     optimizeFocal,
    double*                                  focalInOut,
    double                                   exprTemporalWeight
)
{
    if (shapeBasis.cols() < kShapeCoefficientCount ||
        shapeSigma.size() < kShapeCoefficientCount ||
        exprBasis.cols() < kExpressionCoefficientCount ||
        exprSigma.size() < kExpressionCoefficientCount)
        throw std::runtime_error("fitPoseAndShapeContour: BFM basis/sigma too small");

    // Depth (full fit) setup: cloud normals for point-to-plane, a model-vertex
    // subsample for correspondence, and √weight for the depth term.
    const bool useDepth = targetCloud && !targetCloud->empty();
    if (depthVertexStride < 1) depthVertexStride = 1;
    const std::vector<Eigen::Vector3d> cloudNormals =
        useDepth ? estimateCloudNormals(*targetCloud, 8)
                 : std::vector<Eigen::Vector3d>{};
    const double sqrtDepthWeight = std::sqrt(depthWeight);

    // Split interior (fixed vertex) vs contour (vertexIndex == -1) observations.
    std::vector<LandmarkObservation> fixed, contour;
    for (const LandmarkObservation& o : observations)
        (o.vertexIndex >= 0 ? fixed : contour).push_back(o);

    double angleAxis[3]   = { initialPose.angleAxis.x(),
                              initialPose.angleAxis.y(),
                              initialPose.angleAxis.z() };
    double translation[3] = { initialPose.translation.x(),
                              initialPose.translation.y(),
                              initialPose.translation.z() };
    double shapeCoefficients[kShapeCoefficientCount]      = {0.0};
    double exprCoefficients[kExpressionCoefficientCount]  = {0.0};
    for (int k = 0; k < kShapeCoefficientCount && k < initialIdentity.size(); ++k)
        shapeCoefficients[k] = initialIdentity(k);         // warm start / personalised id
    for (int j = 0; j < kExpressionCoefficientCount && j < initialExpr.size(); ++j)
        exprCoefficients[j] = initialExpr(j);              // warm start expression

    const int N = static_cast<int>(meanShape.rows());
    const float imgW = 2.0f * intrinsics(0, 2);   // ≈ image width (cx ≈ W/2)

    // Camera focal (fx = fy) as an optimisable parameter (webcam personalise).
    // Seeded from *focalInOut / the intrinsics guess; when not optimising the
    // block is held constant, which is numerically the fixed-K fit.
    double focal = (focalInOut && *focalInOut > 0.0) ? *focalInOut
                                                     : intrinsics(0, 0);
    const double focalInit = focal;   // anchor for the focal prior

    // Face-size normalisation for the reprojection residuals: their pixel
    // magnitude scales with resolution / face size but the L2 priors do not, so
    // without this the fit over-articulates on a high-res frame and under-fits a
    // low-res one. Weighting each residual by (refSize/faceSize)² makes the
    // data-vs-prior balance resolution-free. (Depth residuals are metric mm, left
    // un-weighted.)
    double uMin = 1e30, uMax = -1e30, vMin = 1e30, vMax = -1e30;
    for (const LandmarkObservation& o : observations) {
        uMin = std::min(uMin, o.imagePoint.x()); uMax = std::max(uMax, o.imagePoint.x());
        vMin = std::min(vMin, o.imagePoint.y()); vMax = std::max(vMax, o.imagePoint.y());
    }
    const double faceSize = std::max(1.0, std::hypot(uMax - uMin, vMax - vMin));
    const double reprojW  = std::pow(200.0 / faceSize, 2.0);  // 200 px reference face

    std::cout << "\nStarting pose+shape+EXPR CONTOUR fit: " << fixed.size()
              << " interior + " << contour.size() << " contour points"
              << (useDepth ? " + DEPTH (" + std::to_string(targetCloud->size()) +
                             " pts)" : "")
              << ", reg=" << regularizationWeight << " exprReg=" << exprRegWeight
              << '\n';

    for (int outer = 0; outer < numOuterIterations; ++outer) {
        // (a) reconstruct current shape (identity + expression), pose; project.
        Eigen::VectorXf coeff(kShapeCoefficientCount);
        for (int k = 0; k < kShapeCoefficientCount; ++k)
            coeff(k) = static_cast<float>(shapeCoefficients[k]);
        const Eigen::VectorXf dispFlat = shapeBasis.leftCols(kShapeCoefficientCount) *
            shapeSigma.head(kShapeCoefficientCount).cwiseProduct(coeff);
        Eigen::VectorXf ecoeff(kExpressionCoefficientCount);
        for (int j = 0; j < kExpressionCoefficientCount; ++j)
            ecoeff(j) = static_cast<float>(exprCoefficients[j]);
        const Eigen::VectorXf exprFlat = exprBasis.leftCols(kExpressionCoefficientCount) *
            exprSigma.head(kExpressionCoefficientCount).cwiseProduct(ecoeff);
        Eigen::MatrixX3f shape = meanShape;
        for (int v = 0; v < N; ++v)
            shape.row(v) += (dispFlat.segment(3 * v, 3) +
                             exprFlat.segment(3 * v, 3)).transpose();

        const double angle = std::sqrt(angleAxis[0] * angleAxis[0] +
                                       angleAxis[1] * angleAxis[1] +
                                       angleAxis[2] * angleAxis[2]);
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        if (angle > 1e-12) {
            const Eigen::Vector3d axis(angleAxis[0] / angle, angleAxis[1] / angle,
                                       angleAxis[2] / angle);
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix().cast<float>();
        }
        const Eigen::Vector3f t(static_cast<float>(translation[0]),
                                static_cast<float>(translation[1]),
                                static_cast<float>(translation[2]));

        // Current intrinsics (focal may move between outer iterations when it
        // is being optimised) — used for the correspondence projection.
        Eigen::Matrix3f Kcur = intrinsics;
        Kcur(0, 0) = Kcur(1, 1) = static_cast<float>(focal);

        const Eigen::MatrixX3f Vcam = proj::toCameraFrame(shape, R, t);
        const proj::Pixels     uv   = proj::project(Vcam, Kcur);
        const Eigen::MatrixX3f nCam =
            proj::normalsToCameraFrame(Renderer::computeNormals(shape, triangles), R);

        double sumU = 0.0; int cnt = 0;
        for (int v = 0; v < N; ++v)
            if (Vcam(v, 2) > 1e-3f && uv(v, 0) >= 0) { sumU += uv(v, 0); ++cnt; }
        const double centerU = cnt ? sumU / cnt : intrinsics(0, 2);

        ceres::Problem problem;

        // (b) interior landmarks → reprojection residuals (pose+id+expr), Huber-
        // robust: detectors hallucinate the occluded far side under yaw, and one
        // bad landmark would otherwise drag the whole pose. The Huber delta scales
        // with face size, not image size: an open mouth moves lip/chin landmarks
        // ~5% of the face (which must stay quadratic) while true detector garbage
        // (>half the mouth) is suppressed.
        const double interiorHuber = std::max(0.01 * imgW, 0.05 * faceSize);
        // Same reasoning for the jaw contour (it drops ~as far as the chin
        // when the mouth opens), slightly wider since its correspondences are
        // re-matched and inherently noisier.
        const double contourHuber  = std::max(0.02 * imgW, 0.08 * faceSize);
        // Tracking (identity frozen): bake the constant identity displacement
        // into the functor's mean and DROP the identity parameter block — a
        // constant block still costs its full autodiff-jet width, which with
        // 180 identity coeffs was ~3× the per-residual evaluation.
        const auto bakedMean = [&](int v) -> Eigen::Vector3f {
            return meanShape.row(v).transpose() + dispFlat.segment(3 * v, 3);
        };
        for (const LandmarkObservation& o : fixed) {
            if (o.vertexIndex < 0 || o.vertexIndex >= N) continue;
            ceres::LossFunction* loss =
                new ceres::ScaledLoss(new ceres::HuberLoss(interiorHuber),
                                      reprojW, ceres::TAKE_OWNERSHIP);
            if (optimizeIdentity) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true>,
                        2, 3, 3, kShapeCoefficientCount, kExpressionCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<true>(
                            meanShape.row(o.vertexIndex).transpose(), o.vertexIndex,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    loss,
                    angleAxis, translation, shapeCoefficients, exprCoefficients, &focal);
            } else {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true, false>,
                        2, 3, 3, kExpressionCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<true, false>(
                            bakedMean(o.vertexIndex), o.vertexIndex,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    loss,
                    angleAxis, translation, exprCoefficients, &focal);
            }
        }

        // (c) contour points → nearest projected SILHOUETTE vertex (re-matched
        //     every outer iteration = the sliding-contour correspondence).
        int matched = 0;
        for (const LandmarkObservation& o : contour) {
            const double lu = o.imagePoint.x(), lv = o.imagePoint.y();
            const bool lateral   = std::abs(lu - centerU) > 0.03 * imgW;
            const bool leftSide  = lu < centerU;
            int best = -1; double bestD2 = 1e30;
            for (int v = 0; v < N; ++v) {
                if (Vcam(v, 2) <= 1e-3f) continue;                 // behind camera
                if (std::abs(nCam(v, 2)) > 0.5f) continue;         // not near-edge-on
                const double du = uv(v, 0), dv = uv(v, 1);
                if (du < 0 || dv < 0) continue;
                if (std::abs(dv - lv) > 0.12 * imgW) continue;     // same scanline band
                if (lateral && (du < centerU) != leftSide) continue; // same side
                const double d2 = (du - lu) * (du - lu) + (dv - lv) * (dv - lv);
                if (d2 < bestD2) { bestD2 = d2; best = v; }
            }
            if (best < 0 || std::sqrt(bestD2) > 0.12 * imgW) continue;  // gate outliers
            ++matched;
            // The jaw silhouette is an identity signal (face width/length), not
            // expression: with identity free it drives identity; otherwise the
            // solver would reach a low jawline via the jaw-open mode and wrongly
            // open the mouth. With identity frozen (tracking) it drives expression.
            if (optimizeIdentity) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<false>,
                        2, 3, 3, kShapeCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<false>(
                            meanShape.row(best).transpose(), best,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(contourHuber),
                                          reprojW, ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, shapeCoefficients, &focal);
            } else {
                // Identity frozen (tracking) → identity baked into the mean,
                // no identity block (see interior-landmark comment above).
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true, false>,
                        2, 3, 3, kExpressionCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<true, false>(
                            bakedMean(best), best,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(contourHuber),
                                          reprojW, ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, exprCoefficients, &focal);
            }
        }

        // (c2) depth term (full fit only): nearest model vertex per cloud point,
        //      re-matched each outer iteration (ICP), robust + weighted + trimmed.
        // Depth drives pose + identity only, never expression — letting a dense
        // scan move the expression coeffs makes it absorb hair/neck noise via the
        // jaw-open mode. With identity frozen (tracking) it drives pose only,
        // matching the RGB-only path.
        int depthMatched = 0;
        if (useDepth) {
            // First pass: nearest model vertex + distance per cloud point.
            struct DCorr { int vertex; Eigen::Vector3d target, normal; double d2; };
            std::vector<DCorr> dcorr;
            dcorr.reserve(targetCloud->size());
            for (size_t ci = 0; ci < targetCloud->size(); ++ci) {
                const Eigen::Vector3d& tp = (*targetCloud)[ci];
                int best = -1; double bestD2 = std::numeric_limits<double>::max();
                for (int v = 0; v < N; v += depthVertexStride) {
                    const double d2 =
                        (Vcam.row(v).transpose().cast<double>() - tp).squaredNorm();
                    if (d2 < bestD2) { bestD2 = d2; best = v; }
                }
                if (best >= 0) dcorr.push_back({best, tp, cloudNormals[ci], bestD2});
            }
            // Trim the worst 20% by distance — drops hair/neck/off-surface points
            // the face model can never explain, which would otherwise rail the
            // identity coefficients.
            double trimD2 = std::numeric_limits<double>::max();
            if (!dcorr.empty()) {
                std::vector<double> ds; ds.reserve(dcorr.size());
                for (const DCorr& c : dcorr) ds.push_back(c.d2);
                std::sort(ds.begin(), ds.end());
                trimD2 = ds[std::min(ds.size() - 1,
                                     static_cast<size_t>(ds.size() * 0.80))];
            }
            for (const DCorr& c : dcorr) {
                if (c.d2 > trimD2) continue;
                ++depthMatched;
                // Expression displacement of this vertex at the current outer
                // iterate (BFM frame → camera-aligned), constant in the solve.
                const Eigen::Vector3d exprOffset =
                    proj::BFM_TO_CAM.cast<double>() *
                    exprFlat.segment(3 * c.vertex, 3).cast<double>();
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DepthPointResidual,
                        4, 3, 3, kShapeCoefficientCount>(
                        new DepthPointResidual(
                            meanShape.row(c.vertex).transpose(), c.vertex,
                            shapeBasis, shapeSigma,
                            c.target, c.normal, depthPointToPlaneWeight,
                            exprOffset)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(22.0),
                                          sqrtDepthWeight * sqrtDepthWeight,
                                          ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, shapeCoefficients);
            }
        }

        // Identity reg scales with the depth-point count: the depth data term
        // (thousands of mm-scale residuals) grows with it, but a flat prior would
        // then rail the coefficients. Reference 600 points ≈ the un-scaled case.
        const double effRegWeight = useDepth
            ? regularizationWeight * std::max(1.0, depthMatched / 600.0)
            : regularizationWeight;

        // (d) identity (unless frozen for tracking) + expression priors, bounds.
        if (optimizeIdentity) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShapeRegularizationResidual,
                    kShapeCoefficientCount, kShapeCoefficientCount>(
                    new ShapeRegularizationResidual(effRegWeight)),
                nullptr, shapeCoefficients);
            for (int k = 0; k < kShapeCoefficientCount; ++k) {
                problem.SetParameterLowerBound(shapeCoefficients, k, -3.0);
                problem.SetParameterUpperBound(shapeCoefficients, k,  3.0);
            }
        } else if (problem.HasParameterBlock(shapeCoefficients)) {
            problem.SetParameterBlockConstant(shapeCoefficients);   
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoeffPriorResidual<kExpressionCoefficientCount>,
                kExpressionCoefficientCount, kExpressionCoefficientCount>(
                new CoeffPriorResidual<kExpressionCoefficientCount>(exprRegWeight)),
            nullptr, exprCoefficients);
        if (exprTemporalWeight > 0.0 && initialExpr.size() > 0)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CoeffAnchorResidual<kExpressionCoefficientCount>,
                    kExpressionCoefficientCount, kExpressionCoefficientCount>(
                    new CoeffAnchorResidual<kExpressionCoefficientCount>(
                        exprTemporalWeight, initialExpr)),
                nullptr, exprCoefficients);
        if (problem.HasParameterBlock(translation)) {
            problem.SetParameterLowerBound(translation, 2, zMin);
            problem.SetParameterUpperBound(translation, 2, zMax);
        }
        // Wide angle bounds so head rotation (Biwi video) is representable.
        if (problem.HasParameterBlock(angleAxis))
            for (int a = 0; a < 3; ++a) {
                problem.SetParameterLowerBound(angleAxis, a, -1.5);
                problem.SetParameterUpperBound(angleAxis, a,  1.5);
            }
        // Focal: constant unless --optimize-focal (uncalibrated camera). When
        // free: plausible-FOV bounds, a weak anchor at the initial guess, and
        // tight distance bounds around initZ — the subject-distance prior is what
        // disambiguates focal from distance on a single view.
        if (problem.HasParameterBlock(&focal)) {
            if (optimizeFocal) {
                problem.SetParameterLowerBound(&focal, 0, 0.4 * imgW);
                problem.SetParameterUpperBound(&focal, 0, 3.0 * imgW);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<FocalPriorResidual, 1, 1>(
                        new FocalPriorResidual(focalInit, /*weight=*/2.0)),
                    nullptr, &focal);
                if (problem.HasParameterBlock(translation)) {
                    // The caller's distance prior — not initialPose.z, which was
                    // biased by the focal guess. The geometric mean of the z-bounds
                    // (0.4/2.5 × initZ, exact reciprocals) recovers the prior.
                    const double z0 = std::sqrt(zMin * zMax);
                    problem.SetParameterLowerBound(
                        translation, 2, std::max(zMin, 0.75 * z0));
                    problem.SetParameterUpperBound(
                        translation, 2, std::min(zMax, 1.30 * z0));
                }
            } else {
                problem.SetParameterBlockConstant(&focal);
            }
        }
       
        for (int j = 0; j < kExpressionCoefficientCount; ++j) {
            problem.SetParameterLowerBound(exprCoefficients, j, -3.5);
            problem.SetParameterUpperBound(exprCoefficients, j,  3.5);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 60;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "  outer " << outer << " | contour " << matched << "/"
                  << contour.size();
        if (useDepth) std::cout << " | depth " << depthMatched;
        std::cout << " | " << summary.BriefReport() << '\n';
    }

    FitParameters result;
    result.pose.angleAxis =
        Eigen::Vector3d(angleAxis[0], angleAxis[1], angleAxis[2]);
    result.pose.translation =
        Eigen::Vector3d(translation[0], translation[1], translation[2]);
    result.shapeCoefficients.resize(kShapeCoefficientCount);
    for (int k = 0; k < kShapeCoefficientCount; ++k)
        result.shapeCoefficients(k) = shapeCoefficients[k];
    result.exprCoefficients.resize(kExpressionCoefficientCount);
    for (int j = 0; j < kExpressionCoefficientCount; ++j)
        result.exprCoefficients(j) = exprCoefficients[j];
    if (focalInOut) *focalInOut = focal;
    if (optimizeFocal)
        std::cout << "  focal: " << intrinsics(0, 0) << " → " << focal << " px\n";
    std::cout << "Contour fit done. translation="
              << result.pose.translation.transpose()
              << "  |id|=" << result.shapeCoefficients.norm()
              << "  |expr|=" << result.exprCoefficients.norm() << '\n';
    return result;
}

BundleResult CeresFitter::fitIdentityBundle(
    const Eigen::MatrixX3f&          meanShape,
    const Eigen::MatrixXf&           shapeBasis,
    const Eigen::VectorXf&           shapeSigma,
    const Eigen::MatrixXf&           exprBasis,
    const Eigen::VectorXf&           exprSigma,
    const Eigen::MatrixX3i&          triangles,
    const std::vector<BundleFrame>&  frames,
    const Eigen::Matrix3f&           intrinsics,
    double                           regularizationWeight,
    double                           exprRegWeight,
    double                           zMin,
    double                           zMax,
    int                              numOuterIterations,
    double                           depthPointToPlaneWeight,
    double                           depthWeight,
    int                              depthVertexStride)
{
    const int F = static_cast<int>(frames.size());
    if (F == 0) throw std::runtime_error("fitIdentityBundle: no keyframes");
    if (depthVertexStride < 1) depthVertexStride = 1;
    const int   N    = static_cast<int>(meanShape.rows());
    const float imgW = 2.0f * intrinsics(0, 2);
    const double sqrtDepthWeight = std::sqrt(depthWeight);
    const double focalFixed = intrinsics(0, 0);

    // Shared identity; per-frame pose + expression. std::vector storage is
    // stable (never resized after init), so Ceres' pointers stay valid.
    std::array<double, kShapeCoefficientCount> id{};
    std::vector<std::array<double, 3>>                           aa(F), tt(F);
    std::vector<std::array<double, kExpressionCoefficientCount>> ex(F);
    double focal = focalFixed;   // shared, held constant (matches contour fit)

    // Per-frame constants: face-size reprojection weight, split interior/contour
    // observations, and (if present) the depth cloud's normals.
    std::vector<double> reprojW(F);
    std::vector<std::vector<LandmarkObservation>> fixed(F), contour(F);
    std::vector<std::vector<Eigen::Vector3d>>     cloudNormals(F);
    int totalDepthPts = 0;
    for (int f = 0; f < F; ++f) {
        for (int a = 0; a < 3; ++a) {
            aa[f][a] = frames[f].initialPose.angleAxis[a];
            tt[f][a] = frames[f].initialPose.translation[a];
        }
        ex[f].fill(0.0);
        double uMin = 1e30, uMax = -1e30, vMin = 1e30, vMax = -1e30;
        for (const LandmarkObservation& o : frames[f].observations) {
            (o.vertexIndex >= 0 ? fixed[f] : contour[f]).push_back(o);
            uMin = std::min(uMin, o.imagePoint.x()); uMax = std::max(uMax, o.imagePoint.x());
            vMin = std::min(vMin, o.imagePoint.y()); vMax = std::max(vMax, o.imagePoint.y());
        }
        const double faceSize = std::max(1.0, std::hypot(uMax - uMin, vMax - vMin));
        reprojW[f] = std::pow(200.0 / faceSize, 2.0);
        if (frames[f].depthCloud && !frames[f].depthCloud->empty()) {
            cloudNormals[f] = estimateCloudNormals(*frames[f].depthCloud, 8);
            totalDepthPts += static_cast<int>(frames[f].depthCloud->size());
        }
    }
    const bool anyDepth = totalDepthPts > 0;

    std::cout << "\nIDENTITY BUNDLE: " << F << " keyframes"
              << (anyDepth ? " (+depth)" : " (RGB)")
              << ", reg=" << regularizationWeight << '\n';

    for (int outer = 0; outer < numOuterIterations; ++outer) {
        ceres::Problem problem;
        int totalContour = 0, totalDepth = 0;

        for (int f = 0; f < F; ++f) {
            // (a) reconstruct this keyframe's shape (shared id + its expression).
            Eigen::VectorXf idc(kShapeCoefficientCount);
            for (int k = 0; k < kShapeCoefficientCount; ++k) idc(k) = float(id[k]);
            const Eigen::VectorXf dispFlat = shapeBasis.leftCols(kShapeCoefficientCount) *
                shapeSigma.head(kShapeCoefficientCount).cwiseProduct(idc);
            Eigen::VectorXf ecoef(kExpressionCoefficientCount);
            for (int j = 0; j < kExpressionCoefficientCount; ++j) ecoef(j) = float(ex[f][j]);
            const Eigen::VectorXf exprFlat = exprBasis.leftCols(kExpressionCoefficientCount) *
                exprSigma.head(kExpressionCoefficientCount).cwiseProduct(ecoef);
            Eigen::MatrixX3f shape = meanShape;
            for (int v = 0; v < N; ++v)
                shape.row(v) += (dispFlat.segment(3 * v, 3) +
                                 exprFlat.segment(3 * v, 3)).transpose();

            const double ang = std::sqrt(aa[f][0]*aa[f][0]+aa[f][1]*aa[f][1]+aa[f][2]*aa[f][2]);
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            if (ang > 1e-12) {
                const Eigen::Vector3d ax(aa[f][0]/ang, aa[f][1]/ang, aa[f][2]/ang);
                R = Eigen::AngleAxisd(ang, ax).toRotationMatrix().cast<float>();
            }
            const Eigen::Vector3f t(static_cast<float>(tt[f][0]),
                                    static_cast<float>(tt[f][1]),
                                    static_cast<float>(tt[f][2]));
            const Eigen::MatrixX3f Vcam = proj::toCameraFrame(shape, R, t);
            const proj::Pixels     uv   = proj::project(Vcam, intrinsics);
            const Eigen::MatrixX3f nCam =
                proj::normalsToCameraFrame(Renderer::computeNormals(shape, triangles), R);
            double sumU = 0.0; int cnt = 0;
            for (int v = 0; v < N; ++v)
                if (Vcam(v, 2) > 1e-3f && uv(v, 0) >= 0) { sumU += uv(v, 0); ++cnt; }
            const double centerU = cnt ? sumU / cnt : intrinsics(0, 2);

            const double interiorHuber = std::max(0.01 * imgW, 0.05 * (200.0 / std::sqrt(reprojW[f])));
            const double contourHuber  = std::max(0.02 * imgW, 0.08 * (200.0 / std::sqrt(reprojW[f])));

            // (b) interior landmarks → pose(f)+id+expr(f)+focal.
            for (const LandmarkObservation& o : fixed[f]) {
                if (o.vertexIndex < 0 || o.vertexIndex >= N) continue;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true>,
                        2, 3, 3, kShapeCoefficientCount, kExpressionCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<true>(
                            meanShape.row(o.vertexIndex).transpose(), o.vertexIndex,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(interiorHuber),
                                          reprojW[f], ceres::TAKE_OWNERSHIP),
                    aa[f].data(), tt[f].data(), id.data(), ex[f].data(), &focal);
            }

            // (c) jaw contour → nearest projected silhouette vertex → IDENTITY.
            for (const LandmarkObservation& o : contour[f]) {
                const double lu = o.imagePoint.x(), lv = o.imagePoint.y();
                const bool lateral  = std::abs(lu - centerU) > 0.03 * imgW;
                const bool leftSide = lu < centerU;
                int best = -1; double bestD2 = 1e30;
                for (int v = 0; v < N; ++v) {
                    if (Vcam(v, 2) <= 1e-3f) continue;
                    if (std::abs(nCam(v, 2)) > 0.5f) continue;
                    const double du = uv(v, 0), dv = uv(v, 1);
                    if (du < 0 || dv < 0) continue;
                    if (std::abs(dv - lv) > 0.12 * imgW) continue;
                    if (lateral && (du < centerU) != leftSide) continue;
                    const double d2 = (du-lu)*(du-lu) + (dv-lv)*(dv-lv);
                    if (d2 < bestD2) { bestD2 = d2; best = v; }
                }
                if (best < 0 || std::sqrt(bestD2) > 0.12 * imgW) continue;
                ++totalContour;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<false>,
                        2, 3, 3, kShapeCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<false>(
                            meanShape.row(best).transpose(), best,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(contourHuber),
                                          reprojW[f], ceres::TAKE_OWNERSHIP),
                    aa[f].data(), tt[f].data(), id.data(), &focal);
            }

            // (c2) depth ICP → pose(f)+identity (expression baked constant).
            if (frames[f].depthCloud && !frames[f].depthCloud->empty()) {
                const std::vector<Eigen::Vector3d>& cloud = *frames[f].depthCloud;
                struct DCorr { int vertex; Eigen::Vector3d target, normal; double d2; };
                std::vector<DCorr> dcorr; dcorr.reserve(cloud.size());
                for (size_t ci = 0; ci < cloud.size(); ++ci) {
                    const Eigen::Vector3d& tp = cloud[ci];
                    int best = -1; double bestD2 = std::numeric_limits<double>::max();
                    for (int v = 0; v < N; v += depthVertexStride) {
                        const double d2 =
                            (Vcam.row(v).transpose().cast<double>() - tp).squaredNorm();
                        if (d2 < bestD2) { bestD2 = d2; best = v; }
                    }
                    if (best >= 0) dcorr.push_back({best, tp, cloudNormals[f][ci], bestD2});
                }
                double trimD2 = std::numeric_limits<double>::max();
                if (!dcorr.empty()) {
                    std::vector<double> ds; ds.reserve(dcorr.size());
                    for (const DCorr& c : dcorr) ds.push_back(c.d2);
                    std::sort(ds.begin(), ds.end());
                    trimD2 = ds[std::min(ds.size()-1, size_t(ds.size()*0.80))];
                }
                for (const DCorr& c : dcorr) {
                    if (c.d2 > trimD2) continue;
                    ++totalDepth;
                    const Eigen::Vector3d exprOffset = proj::BFM_TO_CAM.cast<double>() *
                        exprFlat.segment(3 * c.vertex, 3).cast<double>();
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<DepthPointResidual,
                            4, 3, 3, kShapeCoefficientCount>(
                            new DepthPointResidual(
                                meanShape.row(c.vertex).transpose(), c.vertex,
                                shapeBasis, shapeSigma,
                                c.target, c.normal, depthPointToPlaneWeight, exprOffset)),
                        new ceres::ScaledLoss(new ceres::HuberLoss(22.0),
                                              sqrtDepthWeight * sqrtDepthWeight,
                                              ceres::TAKE_OWNERSHIP),
                        aa[f].data(), tt[f].data(), id.data());
                }
            }

            // per-frame expression prior + bounds.
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CoeffPriorResidual<kExpressionCoefficientCount>,
                    kExpressionCoefficientCount, kExpressionCoefficientCount>(
                    new CoeffPriorResidual<kExpressionCoefficientCount>(exprRegWeight)),
                nullptr, ex[f].data());
            for (int j = 0; j < kExpressionCoefficientCount; ++j) {
                problem.SetParameterLowerBound(ex[f].data(), j, -3.5);
                problem.SetParameterUpperBound(ex[f].data(), j,  3.5);
            }
            if (problem.HasParameterBlock(tt[f].data())) {
                problem.SetParameterLowerBound(tt[f].data(), 2, zMin);
                problem.SetParameterUpperBound(tt[f].data(), 2, zMax);
            }
            if (problem.HasParameterBlock(aa[f].data()))
                for (int a = 0; a < 3; ++a) {
                    problem.SetParameterLowerBound(aa[f].data(), a, -1.5);
                    problem.SetParameterUpperBound(aa[f].data(), a,  1.5);
                }
        }

        // Shared identity prior. With depth, scale by the (per-frame-average)
        // point count to prevent railing; without depth keep it fixed and low —
        // multi-view landmark/contour parallax over F frames is what safely
        // constrains α (paper w_reg≈0).
        const double effReg = anyDepth
            ? regularizationWeight *
                  std::max(1.0, (totalDepth / static_cast<double>(F)) / 600.0)
            : regularizationWeight;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ShapeRegularizationResidual,
                kShapeCoefficientCount, kShapeCoefficientCount>(
                new ShapeRegularizationResidual(effReg)),
            nullptr, id.data());
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            problem.SetParameterLowerBound(id.data(), k, -3.0);
            problem.SetParameterUpperBound(id.data(), k,  3.0);
        }
        if (problem.HasParameterBlock(&focal))
            problem.SetParameterBlockConstant(&focal);   // intrinsics fixed here

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 40;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double idNorm = 0.0; for (double v : id) idNorm += v * v;
        std::cout << "  bundle outer " << outer << " | contour " << totalContour
                  << (anyDepth ? " | depth " + std::to_string(totalDepth) : "")
                  << " | |id|=" << std::sqrt(idNorm)
                  << " | " << summary.BriefReport() << '\n';
    }

    BundleResult out;
    out.identity.resize(kShapeCoefficientCount);
    for (int k = 0; k < kShapeCoefficientCount; ++k) out.identity(k) = id[k];
    out.poses.resize(F); out.exprs.resize(F);
    for (int f = 0; f < F; ++f) {
        out.poses[f].angleAxis   = Eigen::Vector3d(aa[f][0], aa[f][1], aa[f][2]);
        out.poses[f].translation = Eigen::Vector3d(tt[f][0], tt[f][1], tt[f][2]);
        out.exprs[f].resize(kExpressionCoefficientCount);
        for (int j = 0; j < kExpressionCoefficientCount; ++j) out.exprs[f](j) = ex[f][j];
    }
    std::cout << "Identity bundle done. |id|=" << out.identity.norm() << '\n';
    return out;
}

// Estimate a per-point surface normal via k-nearest-neighbour PCA (smallest
// eigenvector of the local covariance), oriented towards the camera (−Z).
// Computed once before the ICP loop for the point-to-plane term.
static std::vector<Eigen::Vector3d> estimateCloudNormals(
    const std::vector<Eigen::Vector3d>& cloud, int k)
{
    const int n = static_cast<int>(cloud.size());
    std::vector<Eigen::Vector3d> normals(n, Eigen::Vector3d(0.0, 0.0, -1.0));
    k = std::min(k, n - 1);
    if (k < 3) return normals;

    std::vector<std::pair<double, int>> dist(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            dist[j] = { (cloud[j] - cloud[i]).squaredNorm(), static_cast<int>(j) };
        std::partial_sort(dist.begin(), dist.begin() + k + 1, dist.end());

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int m = 1; m <= k; ++m) centroid += cloud[dist[m].second];
        centroid /= static_cast<double>(k);

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int m = 1; m <= k; ++m) {
            const Eigen::Vector3d e = cloud[dist[m].second] - centroid;
            cov += e * e.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        Eigen::Vector3d normal = solver.eigenvectors().col(0);  // smallest eigenvalue
        if (normal.z() > 0.0) normal = -normal;                 // towards camera (−Z)
        normals[i] = normal.normalized();
    }
    return normals;
}

FitParameters CeresFitter::fitDense(
    const Eigen::MatrixX3f&              meanShape,
    const Eigen::MatrixXf&               shapeBasis,
    const Eigen::VectorXf&               shapeSigma,
    const std::vector<Eigen::Vector3d>&  targetCloud,
    const PoseParameters&                initialPose,
    double                               regularizationWeight,
    int                                  numIcpIterations,
    double                               trimPercentile,
    int                                  vertexStride,
    double                               pointToPlaneWeight,
    const DenseIterationCallback&        onIteration
)
{
    if (shapeBasis.cols() < kShapeCoefficientCount ||
        shapeSigma.size() < kShapeCoefficientCount) {
        throw std::runtime_error("fitDense: BFM-Basis/Sigma zu klein");
    }
    if (targetCloud.empty()) {
        throw std::runtime_error("fitDense: leere Ziel-Punktwolke");
    }
    if (vertexStride < 1) vertexStride = 1;

    // Estimate the target cloud's surface normals once (point-to-plane term).
    const std::vector<Eigen::Vector3d> targetNormals =
        estimateCloudNormals(targetCloud, 8);

    const double kPi = 3.14159265358979323846;

    // parameter blocks (Ceres mutates them in place)
    double angleAxis[3] = {
        initialPose.angleAxis.x(),
        initialPose.angleAxis.y(),
        initialPose.angleAxis.z()
    };
    double translation[3] = {
        initialPose.translation.x(),
        initialPose.translation.y(),
        initialPose.translation.z()
    };
    double shapeCoefficients[kShapeCoefficientCount] = {0.0, 0.0, 0.0, 0.0, 0.0};

    // precompute the model-vertex subsample + BFM_TO_CAM once
    // (pose/shape change each iteration; the axis alignment does not.)
    const Eigen::Matrix3d M = proj::BFM_TO_CAM.cast<double>();

    std::vector<int> sub;
    for (int v = 0; v < meanShape.rows(); v += vertexStride) sub.push_back(v);
    const int S = static_cast<int>(sub.size());

    std::vector<Eigen::Vector3d> meanAligned(S);
    std::vector<std::array<Eigen::Vector3d, kShapeCoefficientCount>> basisAligned(S);
    for (int i = 0; i < S; ++i) {
        const int v = sub[i];
        meanAligned[i] = M * meanShape.row(v).transpose().cast<double>();
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            const Eigen::Vector3d d(
                shapeBasis(3 * v + 0, k) * shapeSigma(k),
                shapeBasis(3 * v + 1, k) * shapeSigma(k),
                shapeBasis(3 * v + 2, k) * shapeSigma(k));
            basisAligned[i][k] = M * d;
        }
    }

    std::cout << "\nStarting dense ICP fit: " << targetCloud.size()
              << " target points vs " << S << " model vertices (stride "
              << vertexStride << ")\n";

    // outer ICP loop
    for (int icp = 0; icp < numIcpIterations; ++icp) {

        // (a) current pose/shape → camera-frame positions of the subsample vertices
        const double angle = std::sqrt(angleAxis[0] * angleAxis[0] +
                                       angleAxis[1] * angleAxis[1] +
                                       angleAxis[2] * angleAxis[2]);
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) {
            const Eigen::Vector3d axis(angleAxis[0] / angle,
                                       angleAxis[1] / angle,
                                       angleAxis[2] / angle);
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
        const Eigen::Vector3d t(translation[0], translation[1], translation[2]);

        std::vector<Eigen::Vector3d> modelCam(S);
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d local = meanAligned[i];
            for (int k = 0; k < kShapeCoefficientCount; ++k)
                local += basisAligned[i][k] * shapeCoefficients[k];
            modelCam[i] = R * local + t;
        }

        // (b) correspondence: nearest model vertex per target point (brute force)
        struct Corr {
            int vertexIndex; Eigen::Vector3d target; Eigen::Vector3d normal; double dist;
        };
        std::vector<Corr> correspondences;
        correspondences.reserve(targetCloud.size());
        for (size_t ti = 0; ti < targetCloud.size(); ++ti) {
            const Eigen::Vector3d& tp = targetCloud[ti];
            double best = std::numeric_limits<double>::max();
            int bestI = 0;
            for (int i = 0; i < S; ++i) {
                const double d2 = (modelCam[i] - tp).squaredNorm();
                if (d2 < best) { best = d2; bestI = i; }
            }
            correspondences.push_back({sub[bestI], tp, targetNormals[ti], std::sqrt(best)});
        }

        // (c) trim: drop the worst (100 − trimPercentile)%
        std::vector<double> sorted;
        sorted.reserve(correspondences.size());
        for (const Corr& c : correspondences) sorted.push_back(c.dist);
        std::sort(sorted.begin(), sorted.end());
        const size_t cut = std::min(sorted.size() - 1,
            static_cast<size_t>(sorted.size() * trimPercentile / 100.0));
        const double threshold = sorted[cut];

        // (d) rebuild the Ceres problem from the kept pairs
        ceres::Problem problem;
        int kept = 0;
        double sumSquared = 0.0;
        for (const Corr& c : correspondences) {
            if (c.dist > threshold) continue;
            ++kept;
            sumSquared += c.dist * c.dist;

            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<DepthPointResidual, 4, 3, 3,
                                                kShapeCoefficientCount>(
                    new DepthPointResidual(
                        meanShape.row(c.vertexIndex).transpose(),
                        c.vertexIndex, shapeBasis, shapeSigma,
                        c.target, c.normal, pointToPlaneWeight));

            problem.AddResidualBlock(cost,
                                     new ceres::HuberLoss(10.0),  // robust to outliers
                                     angleAxis, translation, shapeCoefficients);
        }

        // ein Regularisierer (wiederverwendet aus dem Sparse-Fit)
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ShapeRegularizationResidual,
                                            kShapeCoefficientCount,
                                            kShapeCoefficientCount>(
                new ShapeRegularizationResidual(regularizationWeight)),
            nullptr, shapeCoefficients);

        // Bounds: allow the face up to ~3 m → wider z-range than the sparse fit
        problem.SetParameterLowerBound(translation, 2, 100.0);
        problem.SetParameterUpperBound(translation, 2, 3000.0);
        for (int a = 0; a < 3; ++a) {
            problem.SetParameterLowerBound(angleAxis, a, -kPi);
            problem.SetParameterUpperBound(angleAxis, a,  kPi);
        }
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            problem.SetParameterLowerBound(shapeCoefficients, k, -3.0);
            problem.SetParameterUpperBound(shapeCoefficients, k,  3.0);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const double rmse = (kept > 0) ? std::sqrt(sumSquared / kept) : 0.0;
        std::cout << "  icp " << icp << " | kept " << kept << "/"
                  << correspondences.size() << " | RMSE(before solve) "
                  << rmse << " mm | " << summary.BriefReport() << '\n';

        // Fortschritts-Hook: aktuellen Zustand an den Aufrufer geben (z.B. um
        // pro Iteration ein Overlay-Bild zu rendern).
        if (onIteration) {
            FitParameters current;
            current.pose.angleAxis =
                Eigen::Vector3d(angleAxis[0], angleAxis[1], angleAxis[2]);
            current.pose.translation =
                Eigen::Vector3d(translation[0], translation[1], translation[2]);
            current.shapeCoefficients.resize(kShapeCoefficientCount);
            for (int k = 0; k < kShapeCoefficientCount; ++k)
                current.shapeCoefficients(k) = shapeCoefficients[k];
            onIteration(icp, current, rmse);
        }
    }

    // Ergebnis packen
    FitParameters result;
    result.pose.angleAxis = Eigen::Vector3d(angleAxis[0], angleAxis[1], angleAxis[2]);
    result.pose.translation = Eigen::Vector3d(translation[0], translation[1], translation[2]);
    result.shapeCoefficients.resize(kShapeCoefficientCount);
    for (int k = 0; k < kShapeCoefficientCount; ++k)
        result.shapeCoefficients(k) = shapeCoefficients[k];

    std::cout << "Dense fit done.  translation=" << result.pose.translation.transpose()
              << "  shapeCoeff=" << result.shapeCoefficients.transpose() << '\n';
    return result;
}

// appearance estimation helpers (linear, used by fitPhotometric)

// Reconstruct per-vertex albedo from BFM colour coeffs β, clamped to [0,1].
//   albedo(v) = meanAlbedo(v) + Σ_k colorBasis(3v+·, k)·colorSigma(k)·β_k
static Eigen::MatrixX3f albedoFromBeta(const Eigen::MatrixX3f& meanAlbedo,
                                       const Eigen::MatrixXf&  colorBasis,
                                       const Eigen::VectorXf&  colorSigma,
                                       const Eigen::VectorXd&  beta)
{
    Eigen::MatrixX3f out = meanAlbedo;
    const int Kb = static_cast<int>(beta.size());
    if (Kb == 0) return out;
    Eigen::VectorXf sc = colorSigma.head(Kb).cwiseProduct(beta.cast<float>());
    Eigen::VectorXf disp = colorBasis.leftCols(Kb) * sc;   // (3N)
    for (int v = 0; v < out.rows(); ++v)
        out.row(v) += disp.segment(3 * v, 3).transpose();
    return out.cwiseMax(0.0f).cwiseMin(1.0f);
}

// One visible-vertex sample used by both estimators: its camera-frame normal
// and the colour observed at its projection in the input image.
struct AppearanceSample {
    int vertex;
    Eigen::Vector3d observed;   // RGB [0,1] from the photo
};

// Estimate the 9×3 SH lighting by linear least-squares. With geometry + albedo
// fixed, observed_c(v) = albedo_c(v) · (B(n_v)·γ_c), which is linear in γ_c, so
// each channel is an independent 9×9 normal-equation solve. `lambda` conditions
// the (rank-deficient when few normals) system.
static light::SHCoeffs estimateSHLighting(
    const std::vector<AppearanceSample>& samples,
    const Eigen::MatrixX3f&              normalsCam,
    const Eigen::MatrixX3f&              currentAlbedo,
    double                               lambda)
{
    std::array<Eigen::Matrix<double, 9, 9>, 3> AtA;
    std::array<Eigen::Matrix<double, 9, 1>, 3> Atb;
    for (int c = 0; c < 3; ++c) {
        AtA[c] = lambda * Eigen::Matrix<double, 9, 9>::Identity();
        Atb[c].setZero();
    }
    for (const AppearanceSample& s : samples) {
        const Eigen::Matrix<double, 9, 1> b =
            light::shBasis(normalsCam.row(s.vertex)).cast<double>();
        for (int c = 0; c < 3; ++c) {
            const double a = currentAlbedo(s.vertex, c);
            AtA[c].noalias() += (a * a) * (b * b.transpose());
            Atb[c].noalias() += (a * s.observed[c]) * b;
        }
    }
    light::SHCoeffs sh;
    for (int c = 0; c < 3; ++c)
        sh.col(c) = AtA[c].ldlt().solve(Atb[c]).cast<float>();
    return sh;
}

// Estimate BFM albedo coeffs β by regularised linear least-squares: with lighting
// fixed the predicted colour is linear in β. Regularisation pulls β→0 (toward the
// mean albedo) so it doesn't bake lighting/beard/background into the skin colour.
static Eigen::VectorXd estimateAlbedoCoeffs(
    const std::vector<AppearanceSample>& samples,
    const Eigen::MatrixX3f&              normalsCam,
    const light::SHCoeffs&               sh,
    const Eigen::MatrixX3f&              meanAlbedo,
    const Eigen::MatrixXf&               colorBasis,
    const Eigen::VectorXf&               colorSigma,
    int                                  Kb,
    double                               lambda)
{
    Eigen::MatrixXd AtA = lambda * Eigen::MatrixXd::Identity(Kb, Kb);
    Eigen::VectorXd Atb = Eigen::VectorXd::Zero(Kb);
    Eigen::VectorXd row(Kb);
    for (const AppearanceSample& s : samples) {
        const Eigen::RowVector3f shading =
            light::shBasis(normalsCam.row(s.vertex)).transpose() * sh;  // (1×3)
        for (int c = 0; c < 3; ++c) {
            const double sc = shading(c);
            for (int k = 0; k < Kb; ++k)
                row(k) = sc * colorBasis(3 * s.vertex + c, k) * colorSigma(k);
            const double target = s.observed[c] - sc * meanAlbedo(s.vertex, c);
            AtA.noalias() += row * row.transpose();
            Atb.noalias() += row * target;
        }
    }
    Eigen::VectorXd beta = AtA.ldlt().solve(Atb);
    return beta.cwiseMax(-3.0).cwiseMin(3.0);
}

#ifdef USE_CUDA
// small report from the GPU solve so the log can show before/after cost + iters
struct GpuSolveStats { double costInitial = 0.0, costFinal = 0.0; int iters = 0; };

// Host-side Levenberg–Marquardt driving the CUDA photometric kernels; replaces
// the per-pixel Ceres solve in fitPhotometric when --photo-gpu is set. The
// residual/Jacobian (finite-difference) + normal-equation assembly run on the
// GPU; the tiny nParams×nParams system is solved here with Eigen. Params
// (angleAxis, translation, shapeCoefficients) are updated in place.
static void solvePhotometricGpu(
    const cv::Mat&              rgbImage,   // CV_32FC3 RGB [0,1], H×W row-major
    const Eigen::Matrix3f&      K,
    const std::vector<double>&  gBase,
    const std::vector<double>&  gBasis,
    const std::vector<double>&  gTarget,
    int P, int Kshape, bool optimizePose, bool optimizeShape,
    double sqrtWeight, double huberDelta, double shapeRegWeight,
    double tzMin, double tzMax, double shapeLo, double shapeHi,
    double* angleAxis, double* translation, double* shapeCoefficients,
    bool analytic, int maxIters, GpuSolveStats* stats = nullptr)
{
    const int poseParams = optimizePose ? 6 : 0;
    const int nP = poseParams + (optimizeShape ? Kshape : 0);
    if (nP == 0 || P == 0) return;

    void* h = cudaPhotoPrepare(
        reinterpret_cast<const float*>(rgbImage.data), rgbImage.rows, rgbImage.cols,
        K(0, 0), K(1, 1), K(0, 2), K(1, 2), P, gBase.data(),
        optimizeShape ? gBasis.data() : nullptr, gTarget.data(),
        Kshape, optimizeShape ? 1 : 0);

    double aa[3] = { angleAxis[0], angleAxis[1], angleAxis[2] };
    double t[3]  = { translation[0], translation[1], translation[2] };
    std::vector<double> shape(Kshape);
    for (int k = 0; k < Kshape; ++k) shape[k] = shapeCoefficients[k];

    // δ (active-param order: pose 0..5, then shape) → trial (aa,t,shape) + bounds.
    const auto applyDelta = [&](const Eigen::VectorXd& d,
                                double* aaT, double* tT, std::vector<double>& shT) {
        for (int i = 0; i < 3; ++i) { aaT[i] = aa[i]; tT[i] = t[i]; }
        for (int k = 0; k < Kshape; ++k) shT[k] = shape[k];
        int a = 0;
        if (optimizePose) {
            const Eigen::Vector3d drot(d(a), d(a + 1), d(a + 2)); a += 3;
            const Eigen::Vector3d dt  (d(a), d(a + 1), d(a + 2)); a += 3;
            if (analytic) {
                // δrot is a LOCAL SO(3) update: R_new = R(δrot)·R_cur.
                const Eigen::Vector3d aaCur(aa[0], aa[1], aa[2]);
                const double angCur = aaCur.norm();
                const Eigen::Matrix3d Rcur = angCur > 1e-12
                    ? Eigen::Matrix3d(Eigen::AngleAxisd(angCur, aaCur / angCur))
                    : Eigen::Matrix3d::Identity();
                const double angD = drot.norm();
                const Eigen::Matrix3d Rd = angD > 1e-12
                    ? Eigen::Matrix3d(Eigen::AngleAxisd(angD, drot / angD))
                    : Eigen::Matrix3d::Identity();
                const Eigen::AngleAxisd aaNew(Rd * Rcur);
                const Eigen::Vector3d v = aaNew.angle() * aaNew.axis();
                aaT[0] = v(0); aaT[1] = v(1); aaT[2] = v(2);
            } else {                                   // FD: global angle-axis update
                aaT[0] = aa[0] + drot(0); aaT[1] = aa[1] + drot(1); aaT[2] = aa[2] + drot(2);
            }
            tT[0] = t[0] + dt(0); tT[1] = t[1] + dt(1); tT[2] = t[2] + dt(2);
            if (tT[2] < tzMin) tT[2] = tzMin;
            if (tT[2] > tzMax) tT[2] = tzMax;
        }
        if (optimizeShape)
            for (int k = 0; k < Kshape; ++k) {
                shT[k] = shape[k] + d(a++);
                if (shT[k] < shapeLo) shT[k] = shapeLo;
                if (shT[k] > shapeHi) shT[k] = shapeHi;
            }
    };

    const auto addShapePrior = [&](std::vector<double>& JtJ, std::vector<double>& Jtr,
                                   double& cost, const std::vector<double>& sh) {
        if (!optimizeShape) return;
        for (int k = 0; k < Kshape; ++k) {
            const int idx = poseParams + k;
            JtJ[idx * nP + idx] += shapeRegWeight;
            Jtr[idx]            += shapeRegWeight * sh[k];
            cost                += shapeRegWeight * sh[k] * sh[k];
        }
    };

    std::vector<double> JtJ(nP * nP, 0.0), Jtr(nP, 0.0);
    double cost0 = 0.0;
    cudaPhotoNormalEq(h, aa, t, shape.data(), optimizePose ? 1 : 0, optimizeShape ? 1 : 0,
                      analytic ? 1 : 0, sqrtWeight, huberDelta, JtJ.data(), Jtr.data(), &cost0);
    addShapePrior(JtJ, Jtr, cost0, shape);

    const double costInitial = cost0;   // for the log / csv
    int acceptedSteps = 0;

    double lambda = 1e-3;
    for (int iter = 0; iter < maxIters; ++iter) {
        Eigen::MatrixXd A(nP, nP);
        Eigen::VectorXd g(nP);
        for (int i = 0; i < nP; ++i) {
            g(i) = Jtr[i];
            for (int j = 0; j < nP; ++j) A(i, j) = JtJ[i * nP + j];
        }

        bool accepted = false;
        for (int tryk = 0; tryk < 8 && !accepted; ++tryk) {
            Eigen::MatrixXd Ad = A;
            for (int i = 0; i < nP; ++i) Ad(i, i) += lambda * std::max(A(i, i), 1e-12);
            const Eigen::VectorXd delta = Ad.ldlt().solve(-g);

            double aaT[3], tT[3];
            std::vector<double> shT(Kshape);
            applyDelta(delta, aaT, tT, shT);

            double costT = cudaPhotoCost(h, aaT, tT, shT.data(),
                                         optimizePose ? 1 : 0, optimizeShape ? 1 : 0,
                                         sqrtWeight, huberDelta);
            if (optimizeShape)
                for (int k = 0; k < Kshape; ++k) costT += shapeRegWeight * shT[k] * shT[k];

            if (costT < cost0) {                       // accept → recompute at new point
                for (int i = 0; i < 3; ++i) { aa[i] = aaT[i]; t[i] = tT[i]; }
                for (int k = 0; k < Kshape; ++k) shape[k] = shT[k];
                cost0  = costT;
                lambda = std::max(lambda * 0.3, 1e-7);
                accepted = true;
                ++acceptedSteps;
                std::fill(JtJ.begin(), JtJ.end(), 0.0);
                std::fill(Jtr.begin(), Jtr.end(), 0.0);
                double c2 = 0.0;
                cudaPhotoNormalEq(h, aa, t, shape.data(), optimizePose ? 1 : 0,
                                  optimizeShape ? 1 : 0, analytic ? 1 : 0, sqrtWeight, huberDelta,
                                  JtJ.data(), Jtr.data(), &c2);
                addShapePrior(JtJ, Jtr, c2, shape);
            } else {
                lambda *= 10.0;
                if (lambda > 1e12) { accepted = true; iter = maxIters; }  // give up
            }
        }
    }

    for (int i = 0; i < 3; ++i) { angleAxis[i] = aa[i]; translation[i] = t[i]; }
    for (int k = 0; k < Kshape; ++k) shapeCoefficients[k] = shape[k];
    if (stats) { stats->costInitial = costInitial; stats->costFinal = cost0; stats->iters = acceptedSteps; }
    cudaPhotoDestroy(h);
}
#endif  // USE_CUDA

FitParameters CeresFitter::fitPhotometric(
    const Eigen::MatrixX3f&           meanShape,
    const Eigen::MatrixXf&            shapeBasis,
    const Eigen::VectorXf&            shapeSigma,
    const Eigen::MatrixX3i&           triangles,
    const Eigen::MatrixX3f&           meanAlbedo,
    const Eigen::MatrixXf&            colorBasis,
    const Eigen::VectorXf&            colorSigma,
    const cv::Mat&                    imageBgr,
    const Eigen::Matrix3f&            intrinsics,
    const FitParameters&              initialFit,
    double                            shapeRegWeight,
    double                            albedoRegWeight,
    int                               numIterations,
    int                               pixelStride,
    double                            photometricWeight,
    bool                              optimizeShape,
    bool                              optimizeLighting,
    bool                              optimizeAlbedo,
    bool                              optimizePose,
    const DenseIterationCallback&     onIteration,
    int                               maxImageWidth,
    const std::vector<LandmarkObservation>* landmarks,
    double                            landmarkWeight
)
{
    if (shapeBasis.cols() < kShapeCoefficientCount ||
        shapeSigma.size() < kShapeCoefficientCount) {
        throw std::runtime_error("fitPhotometric: BFM basis/sigma too small");
    }
    if (imageBgr.empty()) {
        throw std::runtime_error("fitPhotometric: empty image");
    }
    if (pixelStride < 1) pixelStride = 1;
    if (maxImageWidth < 32) maxImageWidth = 32;

    // With pose AND shape frozen there is nothing for Ceres to solve — the call
    // is a linear lighting/albedo estimate; skip building the per-pixel problem
    // (and its per-vertex precompute) entirely. This is the realtime tracking
    // fast path.
    const bool solveGeometry = optimizePose || optimizeShape;

    // Render + solve at a capped width: a full-res selfie has millions of covered
    // pixels (far too many residuals). Downscale the image and the intrinsics
    // together so the projection stays consistent; the realtime path calls this
    // per pyramid level.
    const double scale =
        std::min(1.0, static_cast<double>(maxImageWidth) / imageBgr.cols);
    cv::Mat imgScaled;
    cv::resize(imageBgr, imgScaled, cv::Size(), scale, scale, cv::INTER_AREA);
    Eigen::Matrix3f K = intrinsics;
    K(0, 0) *= static_cast<float>(scale);   // fx
    K(1, 1) *= static_cast<float>(scale);   // fy
    K(0, 2) *= static_cast<float>(scale);   // cx
    K(1, 2) *= static_cast<float>(scale);   // cy
    const int H = imgScaled.rows;
    const int W = imgScaled.cols;

    // differentiable input image once (BGR → RGB float [0,1], interleaved)
    cv::Mat rgb;
    cv::cvtColor(imgScaled, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, imageBgr.depth() == CV_8U ? 1.0 / 255.0 : 1.0);

    std::vector<double> imageData(static_cast<size_t>(H) * W * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const cv::Vec3f& px = rgb.at<cv::Vec3f>(y, x);
            const size_t idx = (static_cast<size_t>(y) * W + x) * 3;
            imageData[idx + 0] = px[0];
            imageData[idx + 1] = px[1];
            imageData[idx + 2] = px[2];
        }
    const PhotoGrid   grid(imageData.data(), 0, H, 0, W);
    const PhotoInterp image(grid);

    // parameter blocks (Ceres mutates in place)
    double angleAxis[3] = {
        initialFit.pose.angleAxis.x(),
        initialFit.pose.angleAxis.y(),
        initialFit.pose.angleAxis.z()
    };
    double translation[3] = {
        initialFit.pose.translation.x(),
        initialFit.pose.translation.y(),
        initialFit.pose.translation.z()
    };
    double shapeCoefficients[kShapeCoefficientCount] = {0.0};
    for (int k = 0; k < kShapeCoefficientCount &&
                    k < initialFit.shapeCoefficients.size(); ++k)
        shapeCoefficients[k] = initialFit.shapeCoefficients(k);

    // appearance state (estimated linearly each iteration)
    const int Kb = std::min<int>({kAlbedoCoefficientCount,
                                  static_cast<int>(colorBasis.cols()),
                                  static_cast<int>(colorSigma.size())});
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(std::max(Kb, 0));
    for (int k = 0; k < Kb && k < initialFit.albedoCoefficients.size(); ++k)
        beta(k) = initialFit.albedoCoefficients(k);
    light::SHCoeffs sh = initialFit.sh;

    // Precompute each vertex's BFM→camera-aligned mean + shape basis once (the
    // axis alignment doesn't change per iteration). A pixel's surface point is a
    // barycentric blend of three of these which, since the pose is affine and
    // Σbary = 1, reduces to one effective point. Only when geometry is solved.
    const Eigen::Matrix3d M = proj::BFM_TO_CAM.cast<double>();
    const int N = static_cast<int>(meanShape.rows());
    std::vector<Eigen::Vector3d> vMean;
    std::vector<std::array<Eigen::Vector3d, kShapeCoefficientCount>> vBasis;
    if (solveGeometry) {
        vMean.resize(N);
        vBasis.resize(N);
        for (int v = 0; v < N; ++v) {
            vMean[v] = M * meanShape.row(v).transpose().cast<double>();
            for (int k = 0; k < kShapeCoefficientCount; ++k) {
                const Eigen::Vector3d d(
                    shapeBasis(3 * v + 0, k) * shapeSigma(k),
                    shapeBasis(3 * v + 1, k) * shapeSigma(k),
                    shapeBasis(3 * v + 2, k) * shapeSigma(k));
                vBasis[v][k] = M * d;
            }
        }
    }

    const Renderer renderer(H, W, triangles);
    const double sqrtWeight = std::sqrt(photometricWeight);

    std::cout << "\nStarting PER-PIXEL photometric fit (renderer G-buffer): image "
              << W << "x" << H << " (scale " << scale << "), pixel stride "
              << pixelStride << " | optimize shape=" << optimizeShape
              << " lighting=" << optimizeLighting << " albedo="
              << (optimizeAlbedo && Kb > 0) << " (Kβ=" << Kb << ")\n";

    for (int it = 0; it < numIterations; ++it) {

        // (a) reconstruct the current full shape from the shape coefficients
        Eigen::VectorXf coeff(kShapeCoefficientCount);
        for (int k = 0; k < kShapeCoefficientCount; ++k)
            coeff(k) = static_cast<float>(shapeCoefficients[k]);
        const Eigen::VectorXf scaled =
            shapeSigma.head(kShapeCoefficientCount).cwiseProduct(coeff);
        const Eigen::VectorXf dispFlat =
            shapeBasis.leftCols(kShapeCoefficientCount) * scaled;   // (3N)

        Eigen::MatrixX3f shape = meanShape;
        for (int v = 0; v < shape.rows(); ++v)
            shape.row(v) += dispFlat.segment(3 * v, 3).transpose();

        // (b) current pose
        const double angle = std::sqrt(angleAxis[0] * angleAxis[0] +
                                       angleAxis[1] * angleAxis[1] +
                                       angleAxis[2] * angleAxis[2]);
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        if (angle > 1e-12) {
            const Eigen::Vector3d axis(angleAxis[0] / angle,
                                       angleAxis[1] / angle,
                                       angleAxis[2] / angle);
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix().cast<float>();
        }
        const Eigen::Vector3f t(static_cast<float>(translation[0]),
                                static_cast<float>(translation[1]),
                                static_cast<float>(translation[2]));

        // (b2) sample visible vertices (normal + observed photo colour) for the
        //      linear appearance estimators; every 4th vertex is plenty.
        const Eigen::MatrixX3f normals = Renderer::computeNormals(shape, triangles);
        const Eigen::MatrixX3f normalsCam = proj::normalsToCameraFrame(normals, R);
        const Eigen::MatrixX3f camV = proj::toCameraFrame(shape, R, t);
        const proj::Pixels uv = proj::project(camV, K);
        std::vector<AppearanceSample> samples;
        samples.reserve(N / 4);
        for (int v = 0; v < N; v += 4) {
            if (normalsCam(v, 2) >= 0.0f) continue;           // back-facing
            if (camV(v, 2) <= 1e-3f) continue;                // behind camera
            const float u = uv(v, 0), vpx = uv(v, 1);
            if (u < 1.0f || vpx < 1.0f || u >= W - 2.0f || vpx >= H - 2.0f) continue;
            const cv::Vec3f o = rgb.at<cv::Vec3f>(static_cast<int>(vpx),
                                                  static_cast<int>(u));
            samples.push_back({v, Eigen::Vector3d(o[0], o[1], o[2])});
        }

        // (b3) LINEAR appearance estimation: SH lighting, then albedo β, then SH
        //      again so the two are mutually consistent.
        Eigen::MatrixX3f curAlbedo =
            albedoFromBeta(meanAlbedo, colorBasis, colorSigma, beta);
        if (!samples.empty()) {
            if (optimizeLighting)
                sh = estimateSHLighting(samples, normalsCam, curAlbedo, 1e-2);
            if (optimizeAlbedo && Kb > 0) {
                beta = estimateAlbedoCoeffs(samples, normalsCam, sh, meanAlbedo,
                                            colorBasis, colorSigma, Kb, albedoRegWeight);
                curAlbedo = albedoFromBeta(meanAlbedo, colorBasis, colorSigma, beta);
            }
            if (optimizeLighting && optimizeAlbedo && Kb > 0)
                sh = estimateSHLighting(samples, normalsCam, curAlbedo, 1e-2);
        }

        // (c) RENDER the model with the current appearance → predicted image +
        //     G-buffer (triIdx, bary, mask). This is the differentiable-renderer
        //     output the geometry loss is built on.
        const RenderInput in{
            .shape = shape, .albedo = curAlbedo, .R = R, .t = t, .K = K, .sh = sh };
        const RenderOutput out = renderer.render(in);

        // (d) one residual per covered pixel: its surface point (barycentric
        //     blend of the triangle's 3 vertices) reprojected into the input
        //     image vs the rendered colour at that pixel.
#ifdef USE_CUDA
        // GPU path (--photo-gpu): collect the fixed per-pixel correspondences and
        // solve the geometry step on the device (solvePhotometricGpu). The GPU
        // inner solve doesn't include the joint E_lan anchor, so when the caller
        // supplies that anchor (the personalise identity refinement) we stay on
        // the Ceres path; pose-only photometric solves use the GPU.
        const bool gpuSolve = CeresFitter::usePhotometricGpu && solveGeometry &&
                              !(landmarks && landmarkWeight > 0.0);
        std::vector<double> gBase, gBasis, gTarget;
#endif
        // Gather the covered pixels (grid = pixelStride). If photoSamples > 0,
        // keep a random uniform subset for this iteration, re-drawn each outer
        // iteration so all pixels get used over the fit (stochastic subsampling).
        // photoSamples == 0 keeps every pixel.
        std::vector<cv::Point> pts;
        for (int y = 0; y < H; y += pixelStride)
            for (int x = 0; x < W; x += pixelStride)
                if (out.mask.at<uchar>(y, x) && out.triIdx.at<int>(y, x) >= 0)
                    pts.emplace_back(x, y);
        if (CeresFitter::photoSamples > 0 &&
            static_cast<int>(pts.size()) > CeresFitter::photoSamples) {
            std::mt19937 rng(1234u + static_cast<unsigned>(it));   // varies per outer iter
            std::shuffle(pts.begin(), pts.end(), rng);
            pts.resize(static_cast<size_t>(CeresFitter::photoSamples));
        }

        ceres::Problem problem;
        int used = 0;
        double sumSquared = 0.0;
        for (const cv::Point& pt : pts) {
                const int x = pt.x, y = pt.y;
                const int f = out.triIdx.at<int>(y, x);

                // target = rendered colour at this pixel (render() output).
                const cv::Vec3f rc = out.image.at<cv::Vec3f>(y, x);
                const Eigen::Vector3d target(rc[0], rc[1], rc[2]);

                const cv::Vec3f obs = rgb.at<cv::Vec3f>(y, x);
                sumSquared += (target -
                    Eigen::Vector3d(obs[0], obs[1], obs[2])).squaredNorm();
                ++used;

                if (!solveGeometry) continue;   // RMSE only — nothing to solve

                const cv::Vec3f bw = out.bary.at<cv::Vec3f>(y, x);
                const int i0 = triangles(f, 0);
                const int i1 = triangles(f, 1);
                const int i2 = triangles(f, 2);

                // blend the three vertices' aligned mean + basis by the
                // perspective-correct barycentric weights from the G-buffer.
                const Eigen::Vector3d bMean =
                    bw[0] * vMean[i0] + bw[1] * vMean[i1] + bw[2] * vMean[i2];
                std::array<Eigen::Vector3d, kShapeCoefficientCount> bBasis;
                for (int k = 0; k < kShapeCoefficientCount; ++k)
                    bBasis[k] = bw[0] * vBasis[i0][k] +
                                bw[1] * vBasis[i1][k] +
                                bw[2] * vBasis[i2][k];

#ifdef USE_CUDA
                if (gpuSolve) {
                    if (optimizeShape) {                       // surface = base + basis·shape
                        gBase.push_back(bMean.x()); gBase.push_back(bMean.y()); gBase.push_back(bMean.z());
                        for (int k = 0; k < kShapeCoefficientCount; ++k) {
                            gBasis.push_back(bBasis[k].x());
                            gBasis.push_back(bBasis[k].y());
                            gBasis.push_back(bBasis[k].z());
                        }
                    } else {                                   // pose-only: fold frozen shape in
                        Eigen::Vector3d sp = bMean;
                        for (int k = 0; k < kShapeCoefficientCount; ++k)
                            sp += bBasis[k] * shapeCoefficients[k];
                        gBase.push_back(sp.x()); gBase.push_back(sp.y()); gBase.push_back(sp.z());
                    }
                    gTarget.push_back(target.x()); gTarget.push_back(target.y()); gTarget.push_back(target.z());
                    continue;
                }
#endif
                ceres::CostFunction* cost =
                    new ceres::AutoDiffCostFunction<PhotometricPixelResidual, 3,
                                                    3, 3, kShapeCoefficientCount>(
                        new PhotometricPixelResidual(
                            bMean, bBasis, target, K, image, sqrtWeight));

                problem.AddResidualBlock(cost,
                                         new ceres::HuberLoss(0.1),  // robust to specular/outliers
                                         angleAxis, translation, shapeCoefficients);
            }

        // (d2) joint landmark term (Face2Face E_lan): anchor the shape solve to
        //      the detected interior landmarks so the low-reg dense photometric
        //      can't drift the geometry via shape-from-shading ambiguity. Same
        //      pose+shape blocks as the pixel residuals; observations scale to the
        //      working resolution. Only when the caller opts in (weight > 0).
        if (solveGeometry && landmarks && landmarkWeight > 0.0) {
            for (const LandmarkObservation& o : *landmarks) {
                if (o.vertexIndex < 0 || o.vertexIndex >= meanShape.rows()) continue;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkShapeReprojectionResidual,
                        2, 3, 3, kShapeCoefficientCount>(
                        new LandmarkShapeReprojectionResidual(
                            meanShape.row(o.vertexIndex).transpose(), o.vertexIndex,
                            shapeBasis, shapeSigma, o.imagePoint * scale, K)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(4.0),
                                          landmarkWeight, ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, shapeCoefficients);
            }
        }

        if (used == 0) {
            std::cout << "  photo " << it
                      << " | no covered pixels — stopping\n";
            break;
        }

        std::string solverReport;
        ceres::Solver::Summary summary;
        double costInitial = 0.0, costFinal = 0.0;   // cost before/after this solve
        if (solveGeometry) {
#ifdef USE_CUDA
            if (gpuSolve) {
                const int P = static_cast<int>(gTarget.size() / 3);
                GpuSolveStats st;
                solvePhotometricGpu(
                    rgb, K, gBase, gBasis, gTarget, P, kShapeCoefficientCount,
                    optimizePose, optimizeShape, sqrtWeight, /*huberDelta=*/0.1,
                    shapeRegWeight, /*tzMin=*/100.0, /*tzMax=*/3000.0,
                    /*shapeLo=*/-3.0, /*shapeHi=*/3.0,
                    angleAxis, translation, shapeCoefficients,
                    CeresFitter::photoGpuAnalytic, /*maxIters=*/25, &st);
                costInitial = st.costInitial;
                costFinal   = st.costFinal;
                solverReport = std::string("GPU-LM (")
                             + (CeresFitter::photoGpuAnalytic ? "analytic" : "finite-diff")
                             + ")  pixels " + std::to_string(P)
                             + "  cost " + std::to_string(costInitial)
                             + " -> " + std::to_string(costFinal)
                             + "  iters " + std::to_string(st.iters);
            } else
#endif
            {
            if (optimizeShape) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ShapeRegularizationResidual,
                                                    kShapeCoefficientCount,
                                                    kShapeCoefficientCount>(
                        new ShapeRegularizationResidual(shapeRegWeight)),
                    nullptr, shapeCoefficients);
                for (int k = 0; k < kShapeCoefficientCount; ++k) {
                    problem.SetParameterLowerBound(shapeCoefficients, k, -3.0);
                    problem.SetParameterUpperBound(shapeCoefficients, k,  3.0);
                }
            } else {
                problem.SetParameterBlockConstant(shapeCoefficients);
            }

            // Pose: freeze for video tracking (pose comes from the depth fit),
            // else refine it with the photometric term.
            if (optimizePose) {
                problem.SetParameterLowerBound(translation, 2, 100.0);
                problem.SetParameterUpperBound(translation, 2, 3000.0);
            } else {
                problem.SetParameterBlockConstant(angleAxis);
                problem.SetParameterBlockConstant(translation);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.max_num_iterations = 25;
            options.minimizer_progress_to_stdout = false;
            ceres::Solve(options, &problem, &summary);
            costInitial  = summary.initial_cost;
            costFinal    = summary.final_cost;
            solverReport = summary.BriefReport();
            }
        }

        if (solverReport.empty())
            solverReport = solveGeometry ? summary.BriefReport()
                                         : "linear appearance estimate only";

        const double rmse = std::sqrt(sumSquared / used);   // render−photo RMSE, [0,1]
        std::cout << "  photo " << it << " | pixels " << used
                  << " | render-photo RMSE(before solve) " << rmse
                  << " | " << solverReport << '\n';

        // optional csv for plotting convergence: one row per outer iteration
        // columns: iter,pixels,rmse_before,cost_initial,cost_final
        if (!CeresFitter::photoCsvPath.empty()) {
            std::ofstream csv(CeresFitter::photoCsvPath, std::ios::app);
            if (csv) csv << it << ',' << used << ',' << rmse << ','
                         << costInitial << ',' << costFinal << '\n';
        }

        if (onIteration) {
            FitParameters current;
            current.pose.angleAxis =
                Eigen::Vector3d(angleAxis[0], angleAxis[1], angleAxis[2]);
            current.pose.translation =
                Eigen::Vector3d(translation[0], translation[1], translation[2]);
            current.shapeCoefficients.resize(kShapeCoefficientCount);
            for (int k = 0; k < kShapeCoefficientCount; ++k)
                current.shapeCoefficients(k) = shapeCoefficients[k];
            current.albedoCoefficients = beta;   // estimated this iteration
            current.sh = sh;
            onIteration(it, current, rmse);
        }
    }

    FitParameters result;
    result.pose.angleAxis =
        Eigen::Vector3d(angleAxis[0], angleAxis[1], angleAxis[2]);
    result.pose.translation =
        Eigen::Vector3d(translation[0], translation[1], translation[2]);
    result.shapeCoefficients.resize(kShapeCoefficientCount);
    for (int k = 0; k < kShapeCoefficientCount; ++k)
        result.shapeCoefficients(k) = shapeCoefficients[k];
    result.albedoCoefficients = beta;
    result.sh = sh;

    std::cout << "Photometric fit done.  translation="
              << result.pose.translation.transpose()
              << "\n  shapeCoeff="  << result.shapeCoefficients.transpose()
              << "\n  albedoCoeff=" << result.albedoCoefficients.transpose()
              << "\n  sh(DC rgb)="  << result.sh.row(0) << '\n';
    return result;
}

Eigen::VectorXd CeresFitter::fitIdentityPhotometricBundle(
    const Eigen::MatrixX3f&          meanShape,
    const Eigen::MatrixXf&           shapeBasis,
    const Eigen::VectorXf&           shapeSigma,
    const Eigen::MatrixXf&           exprBasis,
    const Eigen::VectorXf&           exprSigma,
    const Eigen::MatrixX3i&          triangles,
    const Eigen::MatrixX3f&          meanAlbedo,
    const Eigen::MatrixXf&           colorBasis,
    const Eigen::VectorXf&           colorSigma,
    const std::vector<cv::Mat>&      bgrs,
    const std::vector<std::vector<LandmarkObservation>>& observations,
    const std::vector<PoseParameters>&  poses,
    const std::vector<Eigen::VectorXd>& exprs,
    const Eigen::Matrix3f&           intrinsics,
    const Eigen::VectorXd&           alphaInit,
    Eigen::VectorXd&                 betaOut,
    light::SHCoeffs&                 shOut,
    double                           shapeReg,
    double                           albedoReg,
    double                           landmarkWeight,
    int                              numIterations,
    int                              pixelStride,
    int                              maxImageWidth)
{
    const int F = static_cast<int>(bgrs.size());
    Eigen::VectorXd alphaV = alphaInit;
    if (F == 0) return alphaV;
    if (pixelStride < 1) pixelStride = 1;
    const int N = static_cast<int>(meanShape.rows());
    const Eigen::Matrix3d Mm = proj::BFM_TO_CAM.cast<double>();

    // Shared identity basis (camera-aligned), precomputed once.
    std::vector<std::array<Eigen::Vector3d, kShapeCoefficientCount>> vBasis(N);
    for (int v = 0; v < N; ++v)
        for (int k = 0; k < kShapeCoefficientCount; ++k)
            vBasis[v][k] = Mm * Eigen::Vector3d(shapeBasis(3*v+0,k)*shapeSigma(k),
                                                shapeBasis(3*v+1,k)*shapeSigma(k),
                                                shapeBasis(3*v+2,k)*shapeSigma(k));

    // Per-keyframe setup (once): downscale image + K, build the differentiable
    // input image, the neutral+expression base shape, camera-aligned vMean, and
    // a renderer. Pose is a constant parameter block (α is the only free one).
    std::vector<cv::Mat>                        rgbF(F);
    std::vector<int>                            Hf(F), Wf(F);
    std::vector<double>                         scaleF(F);
    std::vector<Eigen::Matrix3f>                Kf(F);
    std::vector<std::vector<double>>            imageData(F);
    std::vector<std::unique_ptr<PhotoGrid>>     grids(F);
    std::vector<std::unique_ptr<PhotoInterp>>   interps(F);
    std::vector<Eigen::MatrixX3f>               exprShape(F);   // mean + expr_f
    std::vector<std::vector<Eigen::Vector3d>>   vMeanF(F, std::vector<Eigen::Vector3d>(N));
    std::vector<std::array<double, 3>>          aaK(F), ttK(F);
    std::vector<Eigen::Matrix3f>                Rf(F);
    std::vector<Eigen::Vector3f>                tf(F);
    std::vector<std::unique_ptr<Renderer>>      renderers(F);

    for (int f = 0; f < F; ++f) {
        const double s = std::min(1.0, static_cast<double>(maxImageWidth) / bgrs[f].cols);
        cv::Mat scaled; cv::resize(bgrs[f], scaled, cv::Size(), s, s, cv::INTER_AREA);
        scaleF[f] = s;
        Kf[f] = intrinsics;
        Kf[f](0,0) *= float(s); Kf[f](1,1) *= float(s);
        Kf[f](0,2) *= float(s); Kf[f](1,2) *= float(s);
        Hf[f] = scaled.rows; Wf[f] = scaled.cols;
        cv::Mat rgb; cv::cvtColor(scaled, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, bgrs[f].depth() == CV_8U ? 1.0/255.0 : 1.0);
        rgbF[f] = rgb;
        imageData[f].resize(size_t(Hf[f]) * Wf[f] * 3);
        for (int y = 0; y < Hf[f]; ++y)
            for (int x = 0; x < Wf[f]; ++x) {
                const cv::Vec3f& px = rgb.at<cv::Vec3f>(y, x);
                const size_t idx = (size_t(y) * Wf[f] + x) * 3;
                imageData[f][idx+0]=px[0]; imageData[f][idx+1]=px[1]; imageData[f][idx+2]=px[2];
            }
        grids[f]   = std::make_unique<PhotoGrid>(imageData[f].data(), 0, Hf[f], 0, Wf[f]);
        interps[f] = std::make_unique<PhotoInterp>(*grids[f]);

        // base shape = mean + expression_f (identity added per iteration via α).
        Eigen::VectorXf ecoef(kExpressionCoefficientCount);
        for (int j = 0; j < kExpressionCoefficientCount; ++j)
            ecoef(j) = (j < exprs[f].size()) ? float(exprs[f](j)) : 0.0f;
        const Eigen::VectorXf exprFlat = exprBasis.leftCols(kExpressionCoefficientCount) *
            exprSigma.head(kExpressionCoefficientCount).cwiseProduct(ecoef);
        exprShape[f] = meanShape;
        for (int v = 0; v < N; ++v)
            exprShape[f].row(v) += exprFlat.segment(3*v, 3).transpose();
        for (int v = 0; v < N; ++v)
            vMeanF[f][v] = Mm * exprShape[f].row(v).transpose().cast<double>();

        Rf[f] = poses[f].rotationMatrix();
        tf[f] = poses[f].translation.cast<float>();
        const Eigen::AngleAxisd aaEig(Rf[f].cast<double>());
        const Eigen::Vector3d av = aaEig.angle() * aaEig.axis();
        aaK[f] = {av.x(), av.y(), av.z()};
        ttK[f] = {poses[f].translation.x(), poses[f].translation.y(), poses[f].translation.z()};
        renderers[f] = std::make_unique<Renderer>(Hf[f], Wf[f], triangles);
    }

    const int Kb = std::min<int>({kAlbedoCoefficientCount,
                                  int(colorBasis.cols()), int(colorSigma.size())});
    Eigen::VectorXd beta = (betaOut.size() == Kb) ? betaOut : Eigen::VectorXd::Zero(std::max(Kb,0));
    light::SHCoeffs sh = light::defaultWhite();
    std::array<double, kShapeCoefficientCount> alpha{};
    for (int k = 0; k < kShapeCoefficientCount && k < alphaV.size(); ++k) alpha[k] = alphaV(k);

    // Multi-frame appearance sample: a visible vertex in one keyframe, with its
    // camera-frame normal (pose-dependent) and observed colour.
    struct BSample { int vertex; Eigen::Vector3f n; Eigen::Vector3d obs; };

    std::cout << "\nPHOTOMETRIC BUNDLE: " << F << " keyframes @ " << maxImageWidth
              << "px, refining shared identity\n";

    for (int it = 0; it < numIterations; ++it) {
        // (1) current shape per keyframe (mean + expr_f + α), normals, samples.
        Eigen::VectorXf ac(kShapeCoefficientCount);
        for (int k = 0; k < kShapeCoefficientCount; ++k) ac(k) = float(alpha[k]);
        const Eigen::VectorXf idFlat = shapeBasis.leftCols(kShapeCoefficientCount) *
            shapeSigma.head(kShapeCoefficientCount).cwiseProduct(ac);

        std::vector<Eigen::MatrixX3f> shapeF(F);
        std::vector<BSample> samples;
        for (int f = 0; f < F; ++f) {
            shapeF[f] = exprShape[f];
            for (int v = 0; v < N; ++v) shapeF[f].row(v) += idFlat.segment(3*v,3).transpose();
            const Eigen::MatrixX3f nCam =
                proj::normalsToCameraFrame(Renderer::computeNormals(shapeF[f], triangles), Rf[f]);
            const Eigen::MatrixX3f camV = proj::toCameraFrame(shapeF[f], Rf[f], tf[f]);
            const proj::Pixels uv = proj::project(camV, Kf[f]);
            for (int v = 0; v < N; v += 4) {
                if (nCam(v,2) >= 0.0f || camV(v,2) <= 1e-3f) continue;
                const float u = uv(v,0), vp = uv(v,1);
                if (u < 1.f || vp < 1.f || u >= Wf[f]-2.f || vp >= Hf[f]-2.f) continue;
                const cv::Vec3f o = rgbF[f].at<cv::Vec3f>(int(vp), int(u));
                samples.push_back({v, nCam.row(v).transpose(),
                                   Eigen::Vector3d(o[0], o[1], o[2])});
            }
        }

        // (2) shared lighting γ + albedo β (multi-frame linear LS; SH→β→SH).
        Eigen::MatrixX3f curAlbedo = albedoFromBeta(meanAlbedo, colorBasis, colorSigma, beta);
        auto estimateSH = [&]() {
            std::array<Eigen::Matrix<double,9,9>,3> AtA;
            std::array<Eigen::Matrix<double,9,1>,3> Atb;
            for (int c = 0; c < 3; ++c) { AtA[c] = 1e-2*Eigen::Matrix<double,9,9>::Identity(); Atb[c].setZero(); }
            for (const BSample& s : samples) {
                const Eigen::Matrix<double,9,1> b = light::shBasis(s.n).cast<double>();
                for (int c = 0; c < 3; ++c) {
                    const double a = curAlbedo(s.vertex, c);
                    AtA[c].noalias() += (a*a)*(b*b.transpose());
                    Atb[c].noalias() += (a*s.obs[c])*b;
                }
            }
            for (int c = 0; c < 3; ++c) sh.col(c) = AtA[c].ldlt().solve(Atb[c]).cast<float>();
        };
        if (!samples.empty()) {
            estimateSH();
            if (Kb > 0) {
                Eigen::MatrixXd AtA = albedoReg * Eigen::MatrixXd::Identity(Kb, Kb);
                Eigen::VectorXd Atb = Eigen::VectorXd::Zero(Kb), row(Kb);
                for (const BSample& s : samples) {
                    const Eigen::RowVector3f shading = light::shBasis(s.n).transpose() * sh;
                    for (int c = 0; c < 3; ++c) {
                        const double sc = shading(c);
                        for (int k = 0; k < Kb; ++k)
                            row(k) = sc * colorBasis(3*s.vertex+c, k) * colorSigma(k);
                        const double target = s.obs[c] - sc * meanAlbedo(s.vertex, c);
                        AtA.noalias() += row * row.transpose();
                        Atb.noalias() += row * target;
                    }
                }
                beta = AtA.ldlt().solve(Atb).cwiseMax(-3.0).cwiseMin(3.0);
                curAlbedo = albedoFromBeta(meanAlbedo, colorBasis, colorSigma, beta);
                estimateSH();
            }
        }

        // (3) joint α solve: dense photometric (E_col) + landmarks (E_lan) from
        //     ALL keyframes, per-frame pose fixed → shared α.
        ceres::Problem problem;
        long usedPix = 0;
        for (int f = 0; f < F; ++f) {
            const RenderInput in{ .shape = shapeF[f], .albedo = curAlbedo,
                .R = Rf[f], .t = tf[f], .K = Kf[f], .sh = sh };
            const RenderOutput out = renderers[f]->render(in);
            for (int y = 0; y < Hf[f]; y += pixelStride)
                for (int x = 0; x < Wf[f]; x += pixelStride) {
                    if (!out.mask.at<uchar>(y,x)) continue;
                    const int tri = out.triIdx.at<int>(y,x);
                    if (tri < 0) continue;
                    const cv::Vec3f rc = out.image.at<cv::Vec3f>(y,x);
                    const Eigen::Vector3d target(rc[0], rc[1], rc[2]);
                    const cv::Vec3f bw = out.bary.at<cv::Vec3f>(y,x);
                    const int i0=triangles(tri,0), i1=triangles(tri,1), i2=triangles(tri,2);
                    const Eigen::Vector3d bMean =
                        bw[0]*vMeanF[f][i0] + bw[1]*vMeanF[f][i1] + bw[2]*vMeanF[f][i2];
                    std::array<Eigen::Vector3d, kShapeCoefficientCount> bBasis;
                    for (int k = 0; k < kShapeCoefficientCount; ++k)
                        bBasis[k] = bw[0]*vBasis[i0][k] + bw[1]*vBasis[i1][k] + bw[2]*vBasis[i2][k];
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<PhotometricPixelResidual, 3,
                            3, 3, kShapeCoefficientCount>(
                            new PhotometricPixelResidual(bMean, bBasis, target, Kf[f],
                                                         *interps[f], 1.0)),
                        new ceres::HuberLoss(0.1),
                        aaK[f].data(), ttK[f].data(), alpha.data());
                    ++usedPix;
                }
            // E_lan anchor (interior landmarks) at the working resolution.
            for (const LandmarkObservation& o : observations[f]) {
                if (o.vertexIndex < 0 || o.vertexIndex >= N) continue;
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkShapeReprojectionResidual,
                        2, 3, 3, kShapeCoefficientCount>(
                        new LandmarkShapeReprojectionResidual(
                            exprShape[f].row(o.vertexIndex).transpose(), o.vertexIndex,
                            shapeBasis, shapeSigma, o.imagePoint * scaleF[f], Kf[f])),
                    new ceres::ScaledLoss(new ceres::HuberLoss(4.0),
                                          landmarkWeight, ceres::TAKE_OWNERSHIP),
                    aaK[f].data(), ttK[f].data(), alpha.data());
            }
            problem.SetParameterBlockConstant(aaK[f].data());
            problem.SetParameterBlockConstant(ttK[f].data());
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ShapeRegularizationResidual,
                kShapeCoefficientCount, kShapeCoefficientCount>(
                new ShapeRegularizationResidual(shapeReg)),
            nullptr, alpha.data());
        for (int k = 0; k < kShapeCoefficientCount; ++k) {
            problem.SetParameterLowerBound(alpha.data(), k, -3.0);
            problem.SetParameterUpperBound(alpha.data(), k,  3.0);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 15;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double idn = 0.0; for (double v : alpha) idn += v*v;
        std::cout << "  photo-bundle it " << it << " | pixels " << usedPix
                  << " | |id|=" << std::sqrt(idn) << " | " << summary.BriefReport() << '\n';
    }

    for (int k = 0; k < kShapeCoefficientCount; ++k) alphaV(k) = alpha[k];
    betaOut = beta; shOut = sh;
    std::cout << "Photometric bundle done. |id|=" << alphaV.norm() << '\n';
    return alphaV;
}