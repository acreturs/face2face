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
#include <iostream>
#include <stdexcept>

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <utility>

// GPU photometric inner solve — off unless --photo-gpu (and a USE_CUDA build).
bool CeresFitter::usePhotometricGpu = false;

#ifdef USE_CUDA
// Launchers implemented in src/render/cuda_photometric.cu (built by nvcc).
extern "C" {
void*  cudaPhotoPrepare(const float* image, int H, int W,
                        float fx, float fy, float cx, float cy,
                        int P, const double* basePoint, const double* bBasis,
                        const double* target, int K, int optimizeShape);
void   cudaPhotoNormalEq(void* h, const double* aa, const double* t, const double* shape,
                         int optimizePose, int optimizeShape,
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

// Dense geometry residual (Eq. 2): point-to-point AND point-to-plane.
// Like LandmarkShapeReprojectionResidual, but the residual is the full 3D
// difference (R·v + t) − targetPoint (3 components, point-to-point) PLUS its
// component along the surface normal (1 component, point-to-plane).
// No projection, no intrinsics needed (we are already in the camera frame).
// targetNormal is constant (estimated outside), like targetPoint.
struct DepthPointResidual {
    DepthPointResidual(
        const Eigen::Vector3f& meanPoint,
        int vertexIndex,
        const Eigen::MatrixXf& shapeBasis,
        const Eigen::VectorXf& shapeSigma,
        const Eigen::Vector3d& targetPoint,
        const Eigen::Vector3d& targetNormal,
        double pointToPlaneWeight
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
            proj::BFM_TO_CAM.cast<double>() * meanPoint.cast<double>();

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

// PER-PIXEL photometric residual, driven by the differentiable renderer's
// G-buffer. For one covered pixel of the rendered image:
//
//   surface point  S = b0·v0 + b1·v1 + b2·v2     (barycentric blend of the
//                                                  covering triangle's vertices)
//   residual = w · ( renderedColour(pixel) − inputImage( project(R·S + t) ) )
//
// The barycentric weights (b0,b1,b2) and the target renderedColour both come
// straight from render(): the weights are the perspective-correct `bary`
// G-buffer, the target is `out.image` at that pixel. Correspondence (which
// triangle covers the pixel) is FIXED per outer iteration — the analysis-by-
// synthesis linearisation, exactly like ICP fixes nearest-neighbour pairs.
//
// Because Σ b = 1 and (R,t) is affine, R·(Σ b_j v_j)+t = Σ b_j (R·v_j+t), so we
// pre-blend the vertices' aligned mean + shape basis into ONE effective point.
// The functor is then a single projected point sampled into the input image; as
// pose/shape move, its projection moves and the sampled colour changes, giving a
// gradient w.r.t. pose+shape. At the linearisation point S projects back to its
// own pixel, so the residual there equals renderedColour − inputColour — the
// true photometric error.
struct PhotometricPixelResidual {
    PhotometricPixelResidual(
        const Eigen::Vector3d&                                      blendedMeanAligned,
        const std::array<Eigen::Vector3d, kShapeCoefficientCount>&  blendedBasisAligned,
        const Eigen::Vector3d&  targetColor,       // rendered RGB [0,1], constant
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
        // samples are clamped to the border by Grid2D, so this never faults.
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

// Landmark reprojection with BOTH identity AND expression coefficients. Four
// parameter blocks: angleAxis(3), translation(3), identity(kShape), expr(kExpr).
//   modelPoint = mean + Σ idBasis·alpha + Σ exprBasis·delta   (then pose+project)
// This is how the contour fit gets an expression gradient: mouth/brow landmarks
// move when delta changes.
// Focal-aware variants used by the contour fit: fx = fy = focal[0] is a
// PARAMETER BLOCK (principal point stays fixed), so the camera focal can be
// solved during personalisation on an uncalibrated webcam. When the focal
// block is held constant these are numerically identical to fixed-intrinsics
// reprojection. `WithExpr` selects whether the expression basis contributes
// (contour→identity routing uses the identity-only variant).
template <bool WithExpr>
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

    // WithExpr=true: blocks angleAxis, translation, id, expr, focal.
    // Ceres calls the overload matching the declared block count, so the
    // unused arity is never instantiated.
    template <typename T>
    bool operator()(const T* const angleAxis, const T* const translation,
                    const T* const idCoeff, const T* const exprCoeff,
                    const T* const focal, T* residuals) const {
        return project(angleAxis, translation, idCoeff, exprCoeff, focal, residuals);
    }
    // WithExpr=false: blocks angleAxis, translation, id, focal
    template <typename T>
    bool operator()(const T* const angleAxis, const T* const translation,
                    const T* const idCoeff, const T* const focal,
                    T* residuals) const {
        return project(angleAxis, translation, idCoeff,
                       static_cast<const T*>(nullptr), focal, residuals);
    }

private:
    double meanX_, meanY_, meanZ_;
    std::array<double, kShapeCoefficientCount>      idX_, idY_, idZ_;
    std::array<double, kExpressionCoefficientCount> exX_, exY_, exZ_;
    double observedU_, observedV_, cx_, cy_;
};

// Weak prior anchoring the focal estimate at its initial guess (relative
// deviation). The focal↔distance ambiguity is nearly flat and slightly tilted
// toward long focal (weak perspective fits noisy landmarks better), so an
// unanchored focal rails at its bound; this keeps it in the guess's basin
// while the perspective signal (face depth variation) applies its correction.
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
            nullptr,
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
    double*                                  focalInOut
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

    // Face-size normalisation for the reprojection residuals. Their pixel
    // magnitude scales with resolution / face size, but the L2 priors do not —
    // so without this the fit over-articulates on a high-res (iPhone) frame and
    // under-fits a low-res (Biwi) one. Weighting each reprojection residual by
    // (refSize / faceSize)² makes the data-vs-prior balance resolution-free.
    // (Depth residuals are metric mm — already scale-free — so left un-weighted.)
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

        // (b) interior landmarks → fixed reprojection residuals (pose+id+expr).
        for (const LandmarkObservation& o : fixed) {
            if (o.vertexIndex < 0 || o.vertexIndex >= N) continue;
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true>,
                    2, 3, 3, kShapeCoefficientCount, kExpressionCoefficientCount, 1>(
                    new LandmarkFocalReprojectionResidual<true>(
                        meanShape.row(o.vertexIndex).transpose(), o.vertexIndex,
                        shapeBasis, shapeSigma, exprBasis, exprSigma,
                        o.imagePoint, intrinsics)),
                new ceres::ScaledLoss(nullptr, reprojW, ceres::TAKE_OWNERSHIP),
                angleAxis, translation, shapeCoefficients, exprCoefficients, &focal);
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
            // The jaw silhouette is an IDENTITY signal (face width/length), not
            // an expression one. When identity is free it drives identity only —
            // otherwise the optimiser reaches a low jawline via the jaw-OPEN
            // expression mode, wrongly opening the mouth. When identity is frozen
            // (video tracking) the contour must drive expression instead.
            if (optimizeIdentity) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<false>,
                        2, 3, 3, kShapeCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<false>(
                            meanShape.row(best).transpose(), best,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(0.02 * imgW),
                                          reprojW, ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, shapeCoefficients, &focal);
            } else {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<LandmarkFocalReprojectionResidual<true>,
                        2, 3, 3, kShapeCoefficientCount, kExpressionCoefficientCount, 1>(
                        new LandmarkFocalReprojectionResidual<true>(
                            meanShape.row(best).transpose(), best,
                            shapeBasis, shapeSigma, exprBasis, exprSigma,
                            o.imagePoint, intrinsics)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(0.02 * imgW),
                                          reprojW, ceres::TAKE_OWNERSHIP),
                    angleAxis, translation, shapeCoefficients, exprCoefficients, &focal);
            }
        }

        // (c2) DEPTH term (full fit only): nearest model vertex per cloud point,
        //      re-matched each outer iteration (ICP). Robust + weighted + trimmed.
        //
        // Depth is a geometry signal for pose + IDENTITY only — never expression
        // (letting a dense scan drive the 30 expression coeffs lets it absorb
        // hair/neck/head-band noise via the jaw-open mode, wrongly opening the
        // mouth). With identity free it drives pose+identity; with identity
        // frozen (tracking) the constant shape block means it drives pose only,
        // and expression is left to the landmarks + jaw contour — matching the
        // RGB-only path so both modes stay consistent.
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
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DepthPointResidual,
                        4, 3, 3, kShapeCoefficientCount>(
                        new DepthPointResidual(
                            meanShape.row(c.vertex).transpose(), c.vertex,
                            shapeBasis, shapeSigma,
                            c.target, c.normal, depthPointToPlaneWeight)),
                    new ceres::ScaledLoss(new ceres::HuberLoss(10.0),   // 10 mm
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
            problem.SetParameterBlockConstant(shapeCoefficients);   // tracking: id fixed
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoeffPriorResidual<kExpressionCoefficientCount>,
                kExpressionCoefficientCount, kExpressionCoefficientCount>(
                new CoeffPriorResidual<kExpressionCoefficientCount>(exprRegWeight)),
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
        // Focal: constant unless explicitly optimised (personalisation on an
        // uncalibrated camera). When free: plausible-FOV bounds, a weak
        // anchor prior at the initial guess, and TIGHT distance bounds around
        // the caller's initZ — the subject-distance prior (arm's length for a
        // webcam) is what disambiguates focal from distance on a single view.
        if (problem.HasParameterBlock(&focal)) {
            if (optimizeFocal) {
                problem.SetParameterLowerBound(&focal, 0, 0.4 * imgW);
                problem.SetParameterUpperBound(&focal, 0, 3.0 * imgW);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<FocalPriorResidual, 1, 1>(
                        new FocalPriorResidual(focalInit, /*weight=*/2.0)),
                    nullptr, &focal);
                if (problem.HasParameterBlock(translation)) {
                    // The caller's distance prior. NOT initialPose.z — Stage 1
                    // ran under the focal GUESS, so its distance is biased the
                    // same way as the focal; the geometric mean of the caller's
                    // z-bounds recovers the prior itself (bounds are 0.4/2.5 ×
                    // initZ — exact reciprocals).
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
            problem.SetParameterLowerBound(exprCoefficients, j, -3.0);
            problem.SetParameterUpperBound(exprCoefficients, j,  3.0);
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
              << "\n  exprCoeff=" << result.exprCoefficients.transpose() << '\n';
    return result;
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

    // ── parameter blocks (Ceres mutates them in place) ──
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

    // ── precompute the model-vertex subsample + BFM_TO_CAM once ──
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

    // ── outer ICP loop ──
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

    // ── Ergebnis packen ──
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

// ── appearance estimation helpers (linear, used by fitPhotometric) ───────────

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

// Estimate BFM albedo coeffs β by regularised linear least-squares. With
// lighting fixed, predicted_c(v) = shading_c(v)·(meanAlbedo_c(v) +
// Σ_k colorBasis(3v+c,k)·σ_k·β_k) is linear in β. Regularisation pulls β→0
// (toward the mean albedo) so it does not bake lighting/beard/background into
// the skin colour.
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
// Host-side Levenberg–Marquardt driving the CUDA photometric kernels. Replaces
// the per-pixel Ceres solve inside fitPhotometric when --photo-gpu is set. The
// per-pixel residual/Jacobian (finite-difference) + normal-equation assembly run
// on the GPU; the tiny nParams×nParams system is solved here with Eigen. Params
// (angleAxis, translation, shapeCoefficients) are updated in place. Correspondence
// buffers are laid out per pixel: gBase = 3·P; gBasis = K·3·P (only when
// optimizeShape); gTarget = 3·P. See GPU_RENDERER.md.
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
    int maxIters)
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
            for (int i = 0; i < 3; ++i) aaT[i] = aa[i] + d(a++);
            for (int i = 0; i < 3; ++i) tT[i]  = t[i]  + d(a++);
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
                      sqrtWeight, huberDelta, JtJ.data(), Jtr.data(), &cost0);
    addShapePrior(JtJ, Jtr, cost0, shape);

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
                std::fill(JtJ.begin(), JtJ.end(), 0.0);
                std::fill(Jtr.begin(), Jtr.end(), 0.0);
                double c2 = 0.0;
                cudaPhotoNormalEq(h, aa, t, shape.data(), optimizePose ? 1 : 0,
                                  optimizeShape ? 1 : 0, sqrtWeight, huberDelta,
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
    int                               maxImageWidth
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

    // ── working resolution ──
    // Render + solve at a capped width: a full-res selfie has millions of
    // covered pixels → far too many residuals. Downscale the image AND the
    // intrinsics together so the projection stays consistent. The realtime path
    // calls this per Gaussian-pyramid level (maxImageWidth = 100, 200, …).
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

    // ── differentiable input image ONCE (BGR → RGB float [0,1], interleaved) ──
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

    // ── parameter blocks (Ceres mutates in place) ──
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

    // ── appearance state (estimated linearly each iteration) ──
    const int Kb = std::min<int>({kAlbedoCoefficientCount,
                                  static_cast<int>(colorBasis.cols()),
                                  static_cast<int>(colorSigma.size())});
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(std::max(Kb, 0));
    for (int k = 0; k < Kb && k < initialFit.albedoCoefficients.size(); ++k)
        beta(k) = initialFit.albedoCoefficients(k);
    light::SHCoeffs sh = initialFit.sh;

    // ── precompute each vertex's BFM_TO_CAM-aligned mean + shape basis ONCE ──
    // (pose/shape change each iteration; this axis alignment does not.) A pixel's
    // surface point is a barycentric blend of three of these, which — because the
    // pose is affine and Σbary = 1 — reduces to one effective mean+basis point.
    // Only needed when geometry is actually solved (53k × 30 — skip otherwise).
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
        // GPU path: collect the fixed per-pixel correspondences and solve the
        // whole geometry step on the device (see solvePhotometricGpu). Only when
        // --photo-gpu is set AND geometry is actually solved.
        const bool gpuSolve = CeresFitter::usePhotometricGpu && solveGeometry;
        std::vector<double> gBase, gBasis, gTarget;
#endif
        ceres::Problem problem;
        int used = 0;
        double sumSquared = 0.0;
        for (int y = 0; y < H; y += pixelStride)
            for (int x = 0; x < W; x += pixelStride) {
                if (!out.mask.at<uchar>(y, x)) continue;
                const int f = out.triIdx.at<int>(y, x);
                if (f < 0) continue;

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

        if (used == 0) {
            std::cout << "  photo " << it
                      << " | no covered pixels — stopping\n";
            break;
        }

        std::string solverReport;
        ceres::Solver::Summary summary;
        if (solveGeometry) {
#ifdef USE_CUDA
            if (gpuSolve) {
                const int P = static_cast<int>(gTarget.size() / 3);
                solvePhotometricGpu(
                    rgb, K, gBase, gBasis, gTarget, P, kShapeCoefficientCount,
                    optimizePose, optimizeShape, sqrtWeight, /*huberDelta=*/0.1,
                    shapeRegWeight, /*tzMin=*/100.0, /*tzMax=*/3000.0,
                    /*shapeLo=*/-3.0, /*shapeHi=*/3.0,
                    angleAxis, translation, shapeCoefficients, /*maxIters=*/25);
                solverReport = "GPU-LM (finite-diff)  pixels " + std::to_string(P);
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
            solverReport = summary.BriefReport();
            }
        }

        if (solverReport.empty()) solverReport = summary.BriefReport();

        const double rmse = std::sqrt(sumSquared / used);   // render−photo RMSE, [0,1]
        std::cout << "  photo " << it << " | pixels " << used
                  << " | render-photo RMSE(before solve) " << rmse << " | "
                  << solverReport << '\n';

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