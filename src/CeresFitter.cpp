#include "CeresFitter.h"
#include "ProjectionUtils.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

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