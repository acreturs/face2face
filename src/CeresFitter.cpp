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
    const PoseParameters& initialPose
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

    // Keep the face in front of the camera.
    problem.SetParameterLowerBound(
        translation,
        2,
        100.0
    );

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
    double regularizationWeight
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

    problem.SetParameterLowerBound(
        translation,
        2,
        200.0
    );

    problem.SetParameterUpperBound(
        translation,
        2,
        600.0
    );

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