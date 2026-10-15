// Helpers for Text2Code applications: generate datain with text2code and gather a
// partitioned solution into global element order. Declared in ExasimSolver.hpp.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::vector<double> read_doubles(const fs::path& file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + file.string());
    std::vector<double> a(fs::file_size(file) / sizeof(double));
    in.read(reinterpret_cast<char*>(a.data()), static_cast<std::streamsize>(a.size() * sizeof(double)));
    return a;
}
} // namespace

void GenerateText2CodeDatain(const std::string& pdeappfile, int ranks, int modelnumber,
                             const std::string& text2code)
{
    const fs::path pdeapp = fs::absolute(pdeappfile);
    const std::string suffix = modelnumber > 0 ? std::to_string(modelnumber) : "";
    const fs::path dir = pdeapp.parent_path();
    const fs::path copy = dir / (".exasim_runtime_" + pdeapp.filename().string());
    {
        std::ifstream in(pdeapp);
        std::ofstream out(copy);
        if (!in || !out)
            throw std::runtime_error("cannot read " + pdeapp.string());
        // Later keys override the file's own values.
        out << in.rdbuf() << "\n"
            << "datapath = \"" << dir.string() << "\";\n"
            << "datainpath = \"datain" << suffix << "\";\n"
            << "dataoutpath = \"dataout" << suffix << "\";\n"
            << "mpiprocs = " << ranks << ";\n"
            << "modelnumber = " << modelnumber << ";\n";
    }
    fs::remove_all(dir / ("datain" + suffix));
    fs::create_directories(dir / ("dataout" + suffix));
    const int err = std::system(("\"" + text2code + "\" \"" + copy.string() + "\"").c_str());
    fs::remove(copy);
    if (err != 0)
        throw std::runtime_error("text2code failed for " + pdeapp.string());
}

bool GatherSolution(const std::string& datainpath, const std::string& dataoutpath,
                    const std::string& field, int nranks, const std::string& outfile)
{
    const fs::path datain = datainpath;
    fs::path dataout = dataoutpath;
    std::string lastcase;   // a physicsparam sweep writes one paramcase_NNNN/ per case
    for (const auto& entry : fs::directory_iterator(dataout))
        if (entry.path().filename().string().rfind("paramcase_", 0) == 0)
            lastcase = std::max(lastcase, entry.path().filename().string());
    if (!lastcase.empty())
        dataout /= lastcase;
    if (!fs::exists(dataout / ("out" + field + "_np0.bin")))
        return false;

    // The owned elements of rank r are the first elempartpts[0] + elempartpts[1] entries
    // of elempart in datain/mesh<r+1>.bin (layout: n, nsize[n], ndims, then the arrays).
    std::vector<std::vector<double>> mesh(nranks), data(nranks);
    std::vector<size_t> offset(nranks), owned(nranks);
    size_t ne = 0;
    for (int r = 0; r < nranks; ++r) {
        const fs::path meshfile = datain / ("mesh" + std::to_string(r + 1) + ".bin");
        mesh[r] = read_doubles(fs::exists(meshfile) ? meshfile : datain / "mesh.bin");
        const auto& m = mesh[r];
        offset[r] = 1 + static_cast<size_t>(m[0] + m[1] + m[2] + m[3] + m[4] + m[5] + m[6] + m[7] + m[8] + m[9]);
        const size_t pts = offset[r] + static_cast<size_t>(m[10]);
        owned[r] = static_cast<size_t>(m[pts] + m[pts + 1]);
        ne += owned[r];
        data[r] = read_doubles(dataout / ("out" + field + "_np" + std::to_string(r) + ".bin"));
    }

    // outudg starts with [npe nc ne] and holds every saved step (the last is used);
    // other fields (outvdg) are raw.
    const bool header = field == "udg";
    const double npe = header ? data[0][0] : read_doubles(dataout / "outudg_np0.bin")[0];
    const size_t block = header ? static_cast<size_t>(npe * data[0][1]) : data[0].size() / owned[0];
    std::vector<double> global(block * ne);
    for (int r = 0; r < nranks; ++r) {
        const double* values = data[r].data() + data[r].size() - block * owned[r];
        for (size_t i = 0; i < owned[r]; ++i)
            std::copy_n(values + i * block, block,
                        global.begin() + static_cast<size_t>(mesh[r][offset[r] + i]) * block);
    }

    const double dims[3] = {npe, block / npe, double(ne)};
    std::ofstream out(outfile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(global.data()), static_cast<std::streamsize>(global.size() * sizeof(double)));
    if (!out)
        throw std::runtime_error("cannot write " + outfile);
    std::cout << "Wrote " << outfile << " (" << dims[0] << " x " << dims[1] << " x " << dims[2] << ")\n";
    return true;
}
