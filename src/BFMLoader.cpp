#include "BFMLoader.h"
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5Attribute.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

//aktuell ist albedo noch ignoriert und immer gleich. Ist ein todo!ja bitte

using RowMat3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

// ── Small helpers to read the three PCA blocks (shape/expression/color) ────

static Eigen::VectorXf readVec(const HighFive::File& f, const std::string& path)
{
    std::vector<float> v;
    f.getDataSet(path).read(v);
    return Eigen::Map<Eigen::VectorXf>(v.data(), v.size());
}

static Eigen::VectorXf readSigma(const HighFive::File& f, const std::string& path)
{
    return readVec(f, path).cwiseSqrt();      // σ = sqrt(variance)
}

static Eigen::MatrixXf readBasis(const HighFive::File& f, const std::string& path)
{
    std::vector<std::vector<float>> b;
    f.getDataSet(path).read(b);
    Eigen::MatrixXf M(b.size(), b.front().size());
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
            M(i, j) = b[i][j];
    return M;
}

// Landmarks live in `metadata/landmarks/json` as a UTF-8 JSON string mapping
// landmark name → 3D coordinate. We pair each name with the nearest vertex
// in the mean shape (which is how the Python loader did it).
static std::vector<BFMLandmark> readLandmarks(const HighFive::File& f,
                                              const Eigen::VectorXf& shape_mean)
{
    std::vector<BFMLandmark> out;

    auto tryParse = [&](const std::string& blob) {
        // Minimal parser for the BFM landmark JSON. The file is a single JSON
        // object {"name": [x, y, z], ...}. We extract names and 3-vectors with
        // a tiny hand-rolled scanner so we don't pull in a JSON dependency.
        const int N = static_cast<int>(shape_mean.size() / 3);
        size_t i = 0;
        while (i < blob.size()) {
            size_t k1 = blob.find('"', i);
            if (k1 == std::string::npos) break;
            size_t k2 = blob.find('"', k1 + 1);
            if (k2 == std::string::npos) break;
            std::string name = blob.substr(k1 + 1, k2 - k1 - 1);
            size_t lb = blob.find('[', k2);
            size_t rb = blob.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) break;
            std::string body = blob.substr(lb + 1, rb - lb - 1);
            float xyz[3] = {0, 0, 0};
            int  idx = 0;
            size_t p = 0;
            while (idx < 3 && p < body.size()) {
                while (p < body.size() && (body[p] == ' ' || body[p] == ',')) ++p;
                size_t q = p;
                while (q < body.size() && body[q] != ',' && body[q] != ' ') ++q;
                if (q > p) {
                    try { xyz[idx++] = std::stof(body.substr(p, q - p)); }
                    catch (...) { break; }
                }
                p = q;
            }
            // Nearest vertex in the mean mesh
            int best = -1;
            float best_d = std::numeric_limits<float>::infinity();
            for (int v = 0; v < N; ++v) {
                const float dx = shape_mean[3*v + 0] - xyz[0];
                const float dy = shape_mean[3*v + 1] - xyz[1];
                const float dz = shape_mean[3*v + 2] - xyz[2];
                const float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best_d) { best_d = d2; best = v; }
            }
            if (best >= 0)
                out.push_back({name, best});
            i = rb + 1;
        }
    };

    // BFM-2017 ships the landmarks under metadata/landmarks/json (a string).
    // Wrap in try/catch — older / nomouth variants occasionally use /text.
    try {
        std::string blob;
        f.getDataSet("metadata/landmarks/json").read(blob);
        tryParse(blob);
    } catch (const HighFive::Exception&) {
        try {
            std::string blob;
            f.getDataSet("metadata/landmarks/text").read(blob);
            tryParse(blob);
        } catch (const HighFive::Exception&) {
            // Truly absent — caller will see landmarks().empty().
        }
    }
    return out;
}

BFMLoader::BFMLoader(const std::string& path)
{
    HighFive::File file(path, HighFive::File::ReadOnly);

    // ── identity (shape) PCA ────────────────────────────────────────────────
    shape_mean  = readVec  (file, "shape/model/mean");
    shape_basis = readBasis(file, "shape/model/pcaBasis");
    shape_std   = readSigma(file, "shape/model/pcaVariance");

    // ── expression PCA (additive, identity-independent) ─────────────────────
    expr_mean  = readVec  (file, "expression/model/mean");
    expr_basis = readBasis(file, "expression/model/pcaBasis");
    expr_std   = readSigma(file, "expression/model/pcaVariance");

    // ── color / albedo PCA ──────────────────────────────────────────────────
    color_mean  = readVec  (file, "color/model/mean");
    color_basis = readBasis(file, "color/model/pcaBasis");
    color_std   = readSigma(file, "color/model/pcaVariance");

    // ── topology ────────────────────────────────────────────────────────────
    std::vector<std::vector<int>> cells;
    file.getDataSet("shape/representer/cells").read(cells);
    triangles.resize(cells.front().size(), 3);
    for (int c = 0; c < triangles.rows(); ++c)
        triangles.row(c) << cells[0][c], cells[1][c], cells[2][c];

    // ── named landmarks (BFM↔dlib correspondence source) ────────────────────
    landmarks_ = readLandmarks(file, shape_mean);
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

// identity + additive, identity-independent expression: V = mean_s + B_s·(α⊙σ_id)
//                                                       + mean_e + B_e·(δ⊙σ_exp)
Eigen::MatrixX3f BFMLoader::shape(const Eigen::VectorXf& alpha,
                                  const Eigen::VectorXf& delta) const
{
    const int ka = alpha.size();
    const int kd = delta.size();
    Eigen::VectorXf v = shape_mean
        + shape_basis.leftCols(ka) * (alpha.array() * shape_std.head(ka).array()).matrix()
        + expr_mean
        + expr_basis .leftCols(kd) * (delta.array() * expr_std .head(kd).array()).matrix();
    return Eigen::Map<RowMat3f>(v.data(), v.size() / 3, 3);
}

//generated!
Eigen::MatrixX3f BFMLoader::albedo() const
{
    Eigen::MatrixX3f c = Eigen::Map<const RowMat3f>(color_mean.data(), color_mean.size() / 3, 3);
    if (c.maxCoeff() > 1.5f) c /= 255.0f;          // some models store 0..255
    return c.cwiseMax(0.0f).cwiseMin(1.0f);        // clamp to [0,1] einfach normalisierung
}

// per-vertex albedo with PCA coefficients applied: C = mean_c + B_c·(β⊙σ_alb)
Eigen::MatrixX3f BFMLoader::albedo(const Eigen::VectorXf& beta) const
{
    const int k = beta.size();
    Eigen::VectorXf v = color_mean
        + color_basis.leftCols(k) * (beta.array() * color_std.head(k).array()).matrix();
    Eigen::MatrixX3f c = Eigen::Map<RowMat3f>(v.data(), v.size() / 3, 3);
    if (c.maxCoeff() > 1.5f) c /= 255.0f;
    return c.cwiseMax(0.0f).cwiseMin(1.0f);
}

int BFMLoader::landmark_index(const std::string& name) const
{
    for (const auto& lm : landmarks_)
        if (lm.name == name) return lm.vertex_idx;
    return -1;
}

// ── BFM h5 introspection ────────────────────────────────────────────────────
// Walks the entire HDF5 tree and prints every dataset (path, shape, dtype)
// plus every group's attributes. At the end, prints a quick summary of what
// this BFMLoader instance currently parses into its Eigen members.

static std::string shapeToStr(const std::vector<size_t>& dims)
{
    std::ostringstream s;
    s << "(";
    for (size_t i = 0; i < dims.size(); ++i) {
        s << dims[i];
        if (i + 1 < dims.size()) s << ", ";
    }
    s << ")";
    return s.str();
}

static std::string dtypeToStr(const HighFive::DataType& dt)
{
    // HighFive doesn't expose a friendly name — fall back to class + size in bytes.
    return dt.string();
}

// Safely list attributes — some HDF5 object handles (notably the root file
// handle in HighFive 2.x) raise H5Aget_num_attrs errors instead of returning
// an empty list. Swallow the exception and pretend there were none.
template <typename Obj>
static std::vector<std::string> safeAttrs(const Obj& o)
{
    try { return o.listAttributeNames(); }
    catch (const HighFive::Exception&) { return {}; }
}

static void walk(const HighFive::Group& g, const std::string& prefix)
{
    for (const auto& name : g.listObjectNames()) {
        const std::string full = prefix + "/" + name;
        const auto type = g.getObjectType(name);

        if (type == HighFive::ObjectType::Group) {
            std::cout << "  [G] " << full << "\n";
            HighFive::Group sub = g.getGroup(name);
            for (const auto& a : safeAttrs(sub))
                std::cout << "       @" << a << "\n";
            walk(sub, full);
        }
        else if (type == HighFive::ObjectType::Dataset) {
            HighFive::DataSet ds = g.getDataSet(name);
            const auto space = ds.getSpace();
            const auto dims  = space.getDimensions();
            const auto dt    = ds.getDataType();
            std::cout << "  [D] " << std::left << std::setw(42) << full
                      << "  shape=" << std::setw(18) << shapeToStr(dims)
                      << "  dtype=" << dtypeToStr(dt) << "\n";
            for (const auto& a : safeAttrs(ds))
                std::cout << "       @" << a << "\n";
        }
    }
}

void BFMLoader::summariseBFM(const std::string& path) const
{
    std::cout << "\n──────── BFM h5 summary ────────\n";
    std::cout << "file: " << path << "\n\n";

    HighFive::File file(path, HighFive::File::ReadOnly);

    // Root attributes (skipped if HDF5 reports the file handle as non-attr-bearing)
    for (const auto& a : safeAttrs(file))
        std::cout << "  @" << a << "  (root attribute)\n";

    walk(file.getGroup("/"), "");

    std::cout << "\n──────── Parsed by BFMLoader ────────\n";
    std::cout << "  shape_mean   : " << shape_mean.size()
              << "  (= " << shape_mean.size() / 3 << " vertices × 3)\n";
    std::cout << "  shape_basis  : " << shape_basis.rows() << " × " << shape_basis.cols()
              << "  (3N × K_id)\n";
    std::cout << "  shape_std    : " << shape_std.size()  << "  (σ_id)\n";
    std::cout << "  expr_mean    : " << expr_mean.size()  << "\n";
    std::cout << "  expr_basis   : " << expr_basis.rows() << " × " << expr_basis.cols()
              << "  (3N × K_exp)\n";
    std::cout << "  expr_std     : " << expr_std.size()   << "  (σ_exp)\n";
    std::cout << "  color_mean   : " << color_mean.size() << "\n";
    std::cout << "  color_basis  : " << color_basis.rows() << " × " << color_basis.cols()
              << "  (3N × K_alb)\n";
    std::cout << "  color_std    : " << color_std.size()   << "  (σ_alb)\n";
    std::cout << "  triangles    : " << triangles.rows()   << " × 3\n";
    std::cout << "  landmarks    : " << landmarks_.size()  << " named points\n";
    if (!landmarks_.empty()) {
        std::cout << "    e.g. " << landmarks_.front().name
                  << " -> vertex " << landmarks_.front().vertex_idx << "\n";
    }
    std::cout << "────────────────────────────────────\n\n";
}

