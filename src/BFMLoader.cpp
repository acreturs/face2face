#include "BFMLoader.h"
#include <highfive/H5File.hpp>
#include <vector>

//aktuell ist albedo noch ignoriert und immer gleich. Ist ein todo!ja bitte

using RowMat3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

BFMLoader::BFMLoader(const std::string& path)
{
    HighFive::File file(path, HighFive::File::ReadOnly);

    std::vector<float> mean, color, var;
    file.getDataSet("shape/model/mean").read(mean);
    file.getDataSet("color/model/mean").read(color);
    file.getDataSet("shape/model/pcaVariance").read(var);
    shape_mean = Eigen::Map<Eigen::VectorXf>(mean.data(), mean.size());
    color_mean = Eigen::Map<Eigen::VectorXf>(color.data(), color.size());
    shape_std  = Eigen::Map<Eigen::VectorXf>(var.data(), var.size()).cwiseSqrt();

    std::vector<std::vector<float>> basis; //nested Vector statt Matrix
    file.getDataSet("shape/model/pcaBasis").read(basis);
    shape_basis.resize(basis.size(), basis.front().size());//ist 3n,199
    for (int i = 0; i < shape_basis.rows(); ++i)
        for (int j = 0; j < shape_basis.cols(); ++j)
            shape_basis(i, j) = basis[i][j];

    std::vector<std::vector<int>> cells;
    file.getDataSet("shape/representer/cells").read(cells);
    triangles.resize(cells.front().size(), 3);
    for (int c = 0; c < triangles.rows(); ++c)
        triangles.row(c) << cells[0][c], cells[1][c], cells[2][c];
}

Eigen::MatrixX3f BFMLoader::mean_shape() const
{
    //shape _mean ist ein flacher Vektor (alles untereinander), aber man braucht ihn als nx3 vektor um dann vertices zu rendern
    return Eigen::Map<const RowMat3f>(shape_mean.data(), shape_mean.size() / 3, 3);
}

Eigen::MatrixX3f BFMLoader::shape(const Eigen::VectorXf& alpha) const
{
    const int k = alpha.size(); //benutzt nur die ersten k modes
    Eigen::VectorXf v = shape_mean
        + shape_basis.leftCols(k) * (alpha.array() * shape_std.head(k).array()).matrix();//nur die ersten K von sigma (shape_std)
    return Eigen::Map<RowMat3f>(v.data(), v.size() / 3, 3); //left col ist ersten k spalten, die restlichen modes fallen weg
}

//generated!
Eigen::MatrixX3f BFMLoader::albedo() const
{
    Eigen::MatrixX3f c = Eigen::Map<const RowMat3f>(color_mean.data(), color_mean.size() / 3, 3);
    if (c.maxCoeff() > 1.5f) c /= 255.0f;          // some models store 0..255 
    return c.cwiseMax(0.0f).cwiseMin(1.0f);        // clamp to [0,1] einfach normalisierung
}
