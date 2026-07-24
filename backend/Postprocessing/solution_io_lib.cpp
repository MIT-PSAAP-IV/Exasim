#include "solution_io_lib.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<int> readIntArrayFromDouble(std::ifstream& in,
                                        int count,
                                        const std::string& fname)
{
    if (count < 0) {
        throw std::runtime_error("Negative array length in: " + fname);
    }

    std::vector<int> values(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        double value = 0.0;
        readDoubles(in, &value, 1, fname);
        values[static_cast<std::size_t>(i)] = static_cast<int>(std::round(value));
    }
    return values;
}

void skipIntArrayFromDouble(std::ifstream& in,
                            int count,
                            const std::string& fname)
{
    if (count <= 0) {
        return;
    }
    const std::streamoff bytes =
        static_cast<std::streamoff>(count) * static_cast<std::streamoff>(sizeof(double));
    in.seekg(bytes, std::ios::cur);
    if (!in) {
        throw std::runtime_error("Failed to skip integer array in: " + fname);
    }
}

void readelempartFile(const std::string& filename,
                      std::vector<int>& elempart,
                      std::vector<int>& elempartpts)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    const std::vector<int> lsize = readIntArrayFromDouble(in, 1, filename);
    const std::vector<int> nsize = readIntArrayFromDouble(in, lsize[0], filename);

    if (lsize[0] < 11 || static_cast<int>(nsize.size()) < 11) {
        throw std::runtime_error(
            "readmeshstruct: unexpected mesh file layout (nsize too short) in " + filename);
    }

    const std::vector<int> ndims = readIntArrayFromDouble(in, nsize[0], filename);
    (void)ndims;

    skipIntArrayFromDouble(in, nsize[1], filename); // facecon
    skipIntArrayFromDouble(in, nsize[2], filename); // eblks
    skipIntArrayFromDouble(in, nsize[3], filename); // fblks
    skipIntArrayFromDouble(in, nsize[4], filename); // nbsd
    skipIntArrayFromDouble(in, nsize[5], filename); // elemsend
    skipIntArrayFromDouble(in, nsize[6], filename); // elemrecv
    skipIntArrayFromDouble(in, nsize[7], filename); // elemsendpts
    skipIntArrayFromDouble(in, nsize[8], filename); // elemrecvpts

    elempart = readIntArrayFromDouble(in, nsize[9], filename);
    elempartpts = readIntArrayFromDouble(in, nsize[10], filename);
}

std::size_t sliceOffset4(int n1, int n2, int ne, int elem, int step)
{
    const std::size_t slice = static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
    return slice *
           (static_cast<std::size_t>(elem) +
            static_cast<std::size_t>(ne) * static_cast<std::size_t>(step));
}

std::size_t idx5(int a,
                 int b,
                 int c,
                 int d,
                 int e,
                 int npe2,
                 int p1,
                 int nc,
                 int ne2)
{
    return static_cast<std::size_t>(a) +
           static_cast<std::size_t>(npe2) *
               (static_cast<std::size_t>(b) +
                static_cast<std::size_t>(p1) *
                    (static_cast<std::size_t>(c) +
                     static_cast<std::size_t>(nc) *
                         (static_cast<std::size_t>(d) +
                          static_cast<std::size_t>(ne2) * static_cast<std::size_t>(e))));
}

std::size_t idx4(int a, int b, int c, int d, int n1, int n2, int n3)
{
    return static_cast<std::size_t>(a) +
           static_cast<std::size_t>(n1) *
               (static_cast<std::size_t>(b) +
                static_cast<std::size_t>(n2) *
                    (static_cast<std::size_t>(c) +
                     static_cast<std::size_t>(n3) * static_cast<std::size_t>(d)));
}

} // namespace

void readDoubles(std::ifstream& in,
                 double* dst,
                 std::size_t count,
                 const std::string& fname)
{
    in.read(reinterpret_cast<char*>(dst),
            static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) {
        throw std::runtime_error("Failed to read doubles from file: " + fname);
    }
}

void writeDoubles(std::ofstream& out,
                  const double* values,
                  std::size_t count,
                  const std::string& fname)
{
    out.write(reinterpret_cast<const char*>(values),
              static_cast<std::streamsize>(count * sizeof(double)));
    if (!out) {
        throw std::runtime_error("Failed to write file " + fname);
    }
}

std::int64_t fileSizeBytes(const std::string& fname)
{
    std::ifstream in(fname, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + fname);
    }

    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to get file size for: " + fname);
    }
    return static_cast<std::int64_t>(size);
}

std::vector<int> parseCSVInts(const std::string& s)
{
    std::vector<int> values;
    std::size_t start = 0;
    while (start < s.size()) {
        std::size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        values.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
    }
    return values;
}

void writearray2file(const std::string& filename, const double* values, int count)
{
    if (count <= 0) {
        return;
    }

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file " + filename);
    }
    writeDoubles(out, values, static_cast<std::size_t>(count), filename);
}

void readelempart(const std::string& base,
                  std::vector<std::vector<int>>& elempart,
                  std::vector<std::vector<int>>& elempartpts,
                  int nprocs)
{
    if (nprocs <= 0) {
        throw std::runtime_error("readelempart: nprocs must be positive");
    }

    elempart.resize(static_cast<std::size_t>(nprocs));
    elempartpts.resize(static_cast<std::size_t>(nprocs));

    for (int p = 0; p < nprocs; ++p) {
        std::ostringstream fname;
        fname << base << p + 1 << ".bin";
        readelempartFile(fname.str(),
                         elempart[static_cast<std::size_t>(p)],
                         elempartpts[static_cast<std::size_t>(p)]);
    }
}

void readsolution(const std::string& base,
                  const std::vector<std::vector<int>>& elempartpts,
                  const std::vector<std::vector<int>>& elempart,
                  std::vector<double>& sol3dGlobal,
                  int nsteps,
                  int stepoffsets,
                  int& n1_out,
                  int& n2_out,
                  int& ne_out)
{
    const int nprocs = static_cast<int>(elempartpts.size());
    if (nprocs <= 0) {
        throw std::runtime_error("elempartpts is empty.");
    }
    if (static_cast<int>(elempart.size()) != nprocs) {
        throw std::runtime_error("elempart size mismatch.");
    }
    if (nsteps <= 0) {
        throw std::runtime_error("nsteps must be positive.");
    }
    if (stepoffsets < 0) {
        throw std::runtime_error("stepoffsets must be nonnegative.");
    }

    std::vector<std::array<std::int64_t, 4>> header(static_cast<std::size_t>(nprocs));
    std::int64_t neTotal = 0;

    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        double hdr[3] = {0, 0, 0};
        readDoubles(in, hdr, 3, fname);

        const std::int64_t n1 = static_cast<std::int64_t>(hdr[0]);
        const std::int64_t n2 = static_cast<std::int64_t>(hdr[1]);
        const std::int64_t n3 = static_cast<std::int64_t>(hdr[2]);
        if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
            throw std::runtime_error("Invalid header in: " + fname);
        }

        const std::int64_t nvalues = n1 * n2 * n3;
        const std::int64_t bytes = fileSizeBytes(fname);
        if (bytes % 8 != 0) {
            throw std::runtime_error("File size not multiple of 8 in: " + fname);
        }

        const std::int64_t ndoubles = bytes / 8;
        if (ndoubles < 3) {
            throw std::runtime_error("File too small in: " + fname);
        }
        if ((ndoubles - 3) % nvalues != 0) {
            throw std::runtime_error("Payload not divisible by N in: " + fname);
        }

        const std::int64_t timesteps = (ndoubles - 3) / nvalues;
        header[static_cast<std::size_t>(r)] = {n1, n2, n3, timesteps};
        neTotal += n3;

        if (elempartpts[static_cast<std::size_t>(r)].size() < 2) {
            throw std::runtime_error("elempartpts[" + std::to_string(r) + "] has fewer than 2 entries.");
        }
        if (static_cast<int>(elempart[static_cast<std::size_t>(r)].size()) < static_cast<int>(n3)) {
            throw std::runtime_error("elempart[" + std::to_string(r) + "] shorter than n3.");
        }
        if (timesteps < static_cast<std::int64_t>(stepoffsets + nsteps)) {
            throw std::runtime_error("Requested steps exceed available timesteps in: " + fname);
        }
    }

    const int n1 = static_cast<int>(header[0][0]);
    const int n2 = static_cast<int>(header[0][1]);

    for (int r = 0; r < nprocs; ++r) {
        if (header[static_cast<std::size_t>(r)][0] != n1 ||
            header[static_cast<std::size_t>(r)][1] != n2) {
            throw std::runtime_error("n1/n2 mismatch across rank files.");
        }
    }

    const int ne = static_cast<int>(neTotal);
    n1_out = n1;
    n2_out = n2;
    ne_out = ne;

    const std::size_t slice = static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
    const std::size_t total = slice * static_cast<std::size_t>(ne) *
                              static_cast<std::size_t>(nsteps);
    sol3dGlobal.assign(total, 0.0);

    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        double hdr[3] = {0, 0, 0};
        readDoubles(in, hdr, 3, fname);

        const int n3 = static_cast<int>(hdr[2]);
        const std::size_t rankValues = slice * static_cast<std::size_t>(n3);

        if (stepoffsets > 0) {
            const std::int64_t skipBytes =
                static_cast<std::int64_t>(stepoffsets) *
                static_cast<std::int64_t>(rankValues) * 8;
            in.seekg(skipBytes, std::ios::cur);
            if (!in) {
                throw std::runtime_error("seekg failed in: " + fname);
            }
        }

        std::vector<double> local(rankValues);
        for (int s = 0; s < nsteps; ++s) {
            readDoubles(in, local.data(), rankValues, fname);

            for (int e = 0; e < n3; ++e) {
                const int globalElem = elempart[static_cast<std::size_t>(r)]
                                                [static_cast<std::size_t>(e)];
                if (globalElem < 0 || globalElem >= ne) {
                    throw std::runtime_error("elempart out of range in rank " + std::to_string(r));
                }

                const double* src = local.data() + static_cast<std::size_t>(e) * slice;
                double* dst = sol3dGlobal.data() + sliceOffset4(n1, n2, ne, globalElem, s);
                std::copy(src, src + slice, dst);
            }
        }
    }
}

std::vector<double> extractSol2D(const std::vector<double>& sol3dnew_flat,
                                 int npe2,
                                 int p1,
                                 int nc,
                                 int ne2,
                                 int ne_z,
                                 const std::vector<int>& i_matlab,
                                 const std::vector<int>& j_matlab)
{
    if (i_matlab.size() != j_matlab.size()) {
        throw std::runtime_error("extractSol2D: i_matlab and j_matlab must have the same length.");
    }

    const std::size_t expected =
        static_cast<std::size_t>(npe2) *
        static_cast<std::size_t>(p1) *
        static_cast<std::size_t>(nc) *
        static_cast<std::size_t>(ne2) *
        static_cast<std::size_t>(ne_z);
    if (sol3dnew_flat.size() != expected) {
        throw std::runtime_error("extractSol2D: sol3dnew_flat has unexpected size.");
    }

    const int nij = static_cast<int>(i_matlab.size());
    std::vector<double> sol2dnew(static_cast<std::size_t>(npe2) *
                                 static_cast<std::size_t>(nc) *
                                 static_cast<std::size_t>(ne2) *
                                 static_cast<std::size_t>(nij));

    for (int k = 0; k < nij; ++k) {
        if (i_matlab[static_cast<std::size_t>(k)] < 1 ||
            i_matlab[static_cast<std::size_t>(k)] > p1) {
            throw std::runtime_error("extractSol2D: i_matlab out of range at entry " +
                                     std::to_string(k));
        }
        if (j_matlab[static_cast<std::size_t>(k)] < 1 ||
            j_matlab[static_cast<std::size_t>(k)] > ne_z) {
            throw std::runtime_error("extractSol2D: j_matlab out of range at entry " +
                                     std::to_string(k));
        }

        const int i = i_matlab[static_cast<std::size_t>(k)] - 1;
        const int j = j_matlab[static_cast<std::size_t>(k)] - 1;

        for (int c = 0; c < ne2; ++c) {
            for (int b = 0; b < nc; ++b) {
                for (int a = 0; a < npe2; ++a) {
                    sol2dnew[idx4(a, b, c, k, npe2, nc, ne2)] =
                        sol3dnew_flat[idx5(a, i, b, c, j, npe2, p1, nc, ne2)];
                }
            }
        }
    }

    return sol2dnew;
}
