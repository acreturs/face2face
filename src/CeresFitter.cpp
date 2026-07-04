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

} // namespace

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
    const DenseIterationCallback&     onIteration
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

    // ── working resolution ──
    // Render + solve at a capped width: a full-res selfie has millions of
    // covered pixels → far too many residuals. Downscale the image AND the
    // intrinsics together so the projection stays consistent.
    constexpr int kMaxWidth = 400;
    const double scale =
        std::min(1.0, static_cast<double>(kMaxWidth) / imageBgr.cols);
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
    const Eigen::Matrix3d M = proj::BFM_TO_CAM.cast<double>();
    const int N = static_cast<int>(meanShape.rows());
    std::vector<Eigen::Vector3d> vMean(N);
    std::vector<std::array<Eigen::Vector3d, kShapeCoefficientCount>> vBasis(N);
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
        ceres::Problem problem;
        int used = 0;
        double sumSquared = 0.0;
        for (int y = 0; y < H; y += pixelStride)
            for (int x = 0; x < W; x += pixelStride) {
                if (!out.mask.at<uchar>(y, x)) continue;
                const int f = out.triIdx.at<int>(y, x);
                if (f < 0) continue;

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

                // target = rendered colour at this pixel (render() output).
                const cv::Vec3f rc = out.image.at<cv::Vec3f>(y, x);
                const Eigen::Vector3d target(rc[0], rc[1], rc[2]);

                const cv::Vec3f obs = rgb.at<cv::Vec3f>(y, x);
                sumSquared += (target -
                    Eigen::Vector3d(obs[0], obs[1], obs[2])).squaredNorm();
                ++used;

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

        problem.SetParameterLowerBound(translation, 2, 100.0);
        problem.SetParameterUpperBound(translation, 2, 3000.0);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 25;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const double rmse = std::sqrt(sumSquared / used);   // render−photo RMSE, [0,1]
        std::cout << "  photo " << it << " | pixels " << used
                  << " | render-photo RMSE(before solve) " << rmse << " | "
                  << summary.BriefReport() << '\n';

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