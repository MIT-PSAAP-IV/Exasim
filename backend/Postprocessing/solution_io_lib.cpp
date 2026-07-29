#include "solution_io_lib.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
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

bool isPositiveIntegerDouble(double value)
{
    return std::isfinite(value) &&
           value > 0.0 &&
           std::floor(value) == value &&
           value <= static_cast<double>(std::numeric_limits<int>::max());
}

std::size_t checkedProduct3(int n1, int n2, int n3, const std::string& context)
{
    const std::size_t a = static_cast<std::size_t>(n1);
    const std::size_t b = static_cast<std::size_t>(n2);
    const std::size_t c = static_cast<std::size_t>(n3);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::runtime_error("Size overflow in " + context);
    }
    const std::size_t ab = a * b;
    if (ab > std::numeric_limits<std::size_t>::max() / c) {
        throw std::runtime_error("Size overflow in " + context);
    }
    return ab * c;
}

std::size_t checkedProduct4(int n1, int n2, int n3, int n4, const std::string& context)
{
    const std::size_t abc = checkedProduct3(n1, n2, n3, context);
    const std::size_t d = static_cast<std::size_t>(n4);
    if (abc > std::numeric_limits<std::size_t>::max() / d) {
        throw std::runtime_error("Size overflow in " + context);
    }
    return abc * d;
}

std::string withBinSuffix(const std::string& base)
{
    const std::string suffix = ".bin";
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return base;
    }
    return base + suffix;
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

void writeFieldWithHeader(const std::string& filename,
                          const std::vector<double>& values,
                          int n1,
                          int n2,
                          int n3)
{
    if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
        throw std::runtime_error("writeFieldWithHeader: dimensions must be positive for " + filename);
    }

    const std::size_t expected = checkedProduct3(n1, n2, n3, filename);
    if (values.size() != expected) {
        throw std::runtime_error("writeFieldWithHeader: payload size does not match header for " +
                                 filename);
    }

    const std::string output = withBinSuffix(filename);
    std::ofstream out(output, std::ios::out | std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file " + output);
    }

    const double hdr[3] = {
        static_cast<double>(n1),
        static_cast<double>(n2),
        static_cast<double>(n3)
    };
    writeDoubles(out, hdr, 3, output);
    writeDoubles(out, values.data(), values.size(), output);
}

void writeFieldWithHeader4(const std::string& filename,
                           const std::vector<double>& values,
                           int n1,
                           int n2,
                           int n3,
                           int n4)
{
    if (n1 <= 0 || n2 <= 0 || n3 <= 0 || n4 <= 0) {
        throw std::runtime_error("writeFieldWithHeader4: dimensions must be positive for " +
                                 filename);
    }

    const std::size_t expected = checkedProduct4(n1, n2, n3, n4, filename);
    if (values.size() != expected) {
        throw std::runtime_error("writeFieldWithHeader4: payload size does not match header for " +
                                 filename);
    }

    const std::string output = withBinSuffix(filename);
    std::ofstream out(output, std::ios::out | std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file " + output);
    }

    const double hdr[4] = {
        static_cast<double>(n1),
        static_cast<double>(n2),
        static_cast<double>(n3),
        static_cast<double>(n4)
    };
    writeDoubles(out, hdr, 4, output);
    writeDoubles(out, values.data(), values.size(), output);
}

void getxf(const std::string& base,
           int nprocs,
           std::vector<double>& xf,
           int& n1_out,
           int& n2_out,
           int& n3_out)
{
    if (nprocs <= 0) {
        throw std::runtime_error("getxf: nprocs must be positive.");
    }

    struct RankHeader {
        int n1 = 0;
        int n2 = 0;
        int n3 = 0;
    };

    std::vector<RankHeader> headers(static_cast<std::size_t>(nprocs));
    int n1Global = 0;
    int n3Global = 0;
    std::int64_t n2Total = 0;

    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        double hdr[3] = {0.0, 0.0, 0.0};
        readDoubles(in, hdr, 3, fname);
        for (int k = 0; k < 3; ++k) {
            if (!isPositiveIntegerDouble(hdr[k])) {
                throw std::runtime_error("Invalid header value in: " + fname);
            }
        }

        const int n1 = static_cast<int>(hdr[0]);
        const int n2 = static_cast<int>(hdr[1]);
        const int n3 = static_cast<int>(hdr[2]);
        const std::size_t nvalues = checkedProduct3(n1, n2, n3, fname);

        const std::int64_t bytes = fileSizeBytes(fname);
        if (bytes % static_cast<std::int64_t>(sizeof(double)) != 0) {
            throw std::runtime_error("File size not multiple of 8 in: " + fname);
        }
        const std::int64_t ndoubles =
            bytes / static_cast<std::int64_t>(sizeof(double));
        if (ndoubles != static_cast<std::int64_t>(3 + nvalues)) {
            throw std::runtime_error("Unexpected payload size in: " + fname);
        }

        if (r == 0) {
            n1Global = n1;
            n3Global = n3;
        } else if (n1 != n1Global || n3 != n3Global) {
            throw std::runtime_error("n1/n3 mismatch across rank files at: " + fname);
        }

        headers[static_cast<std::size_t>(r)] = {n1, n2, n3};
        n2Total += n2;
        if (n2Total > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("getxf: total n2 exceeds int range.");
        }
    }

    const int n2Global = static_cast<int>(n2Total);
    const std::size_t total = checkedProduct3(n1Global, n2Global, n3Global, base);
    xf.assign(total, 0.0);

    int n2Offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        const RankHeader h = headers[static_cast<std::size_t>(r)];
        const std::size_t localSize = checkedProduct3(h.n1, h.n2, h.n3, fname);

        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }
        double hdr[3] = {0.0, 0.0, 0.0};
        readDoubles(in, hdr, 3, fname);

        std::vector<double> local(localSize);
        readDoubles(in, local.data(), local.size(), fname);

        for (int k = 0; k < h.n3; ++k) {
            for (int j = 0; j < h.n2; ++j) {
                const std::size_t src =
                    static_cast<std::size_t>(h.n1) *
                    (static_cast<std::size_t>(j) +
                     static_cast<std::size_t>(h.n2) * static_cast<std::size_t>(k));
                const std::size_t dst =
                    static_cast<std::size_t>(n1Global) *
                    (static_cast<std::size_t>(n2Offset + j) +
                     static_cast<std::size_t>(n2Global) * static_cast<std::size_t>(k));
                std::copy(local.data() + src,
                          local.data() + src + static_cast<std::size_t>(h.n1),
                          xf.data() + dst);
            }
        }

        n2Offset += h.n2;
    }

    n1_out = n1Global;
    n2_out = n2Global;
    n3_out = n3Global;
}

void getudgf(const std::string& base,
             int nprocs,
             int nsteps,
             int stepoffsets,
             std::vector<double>& udgf,
             int& n1_out,
             int& n2_out,
             int& n3_out,
             int& n4_out)
{
    if (nprocs <= 0) {
        throw std::runtime_error("getudgf: nprocs must be positive.");
    }
    if (nsteps <= 0) {
        throw std::runtime_error("getudgf: nsteps must be positive.");
    }
    if (stepoffsets < 0) {
        throw std::runtime_error("getudgf: stepoffsets must be nonnegative.");
    }

    struct RankHeader {
        int n1 = 0;
        int n2 = 0;
        int n3 = 0;
    };

    std::vector<RankHeader> headers(static_cast<std::size_t>(nprocs));
    int n1Global = 0;
    int n3Global = 0;
    std::int64_t n2Total = 0;

    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }

        double hdr[3] = {0.0, 0.0, 0.0};
        readDoubles(in, hdr, 3, fname);
        for (int k = 0; k < 3; ++k) {
            if (!isPositiveIntegerDouble(hdr[k])) {
                throw std::runtime_error("Invalid header value in: " + fname);
            }
        }

        const int n1 = static_cast<int>(hdr[0]);
        const int n2 = static_cast<int>(hdr[1]);
        const int n3 = static_cast<int>(hdr[2]);
        const std::size_t nvalues = checkedProduct3(n1, n2, n3, fname);

        const std::int64_t bytes = fileSizeBytes(fname);
        if (bytes % static_cast<std::int64_t>(sizeof(double)) != 0) {
            throw std::runtime_error("File size not multiple of 8 in: " + fname);
        }
        const std::int64_t ndoubles =
            bytes / static_cast<std::int64_t>(sizeof(double));
        if (ndoubles < 3) {
            throw std::runtime_error("File too small in: " + fname);
        }
        if ((ndoubles - 3) % static_cast<std::int64_t>(nvalues) != 0) {
            throw std::runtime_error("Payload not divisible by snapshot size in: " + fname);
        }

        const std::int64_t availableSteps =
            (ndoubles - 3) / static_cast<std::int64_t>(nvalues);
        if (static_cast<std::int64_t>(stepoffsets) + static_cast<std::int64_t>(nsteps) >
            availableSteps) {
            throw std::runtime_error("Requested steps exceed available timesteps in: " + fname);
        }

        if (r == 0) {
            n1Global = n1;
            n3Global = n3;
        } else if (n1 != n1Global || n3 != n3Global) {
            throw std::runtime_error("n1/n3 mismatch across rank files at: " + fname);
        }

        headers[static_cast<std::size_t>(r)] = {n1, n2, n3};
        n2Total += n2;
        if (n2Total > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("getudgf: total n2 exceeds int range.");
        }
    }

    const int n2Global = static_cast<int>(n2Total);
    const std::size_t total = checkedProduct4(n1Global, n2Global, n3Global, nsteps, base);
    udgf.assign(total, 0.0);

    int n2Offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        const RankHeader h = headers[static_cast<std::size_t>(r)];
        const std::size_t rankValues = checkedProduct3(h.n1, h.n2, h.n3, fname);
        const std::size_t localSize =
            rankValues * static_cast<std::size_t>(nsteps);

        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + fname);
        }
        double hdr[3] = {0.0, 0.0, 0.0};
        readDoubles(in, hdr, 3, fname);

        const std::streamoff skipBytes =
            static_cast<std::streamoff>(stepoffsets) *
            static_cast<std::streamoff>(rankValues) *
            static_cast<std::streamoff>(sizeof(double));
        if (skipBytes > 0) {
            in.seekg(skipBytes, std::ios::cur);
            if (!in) {
                throw std::runtime_error("seekg failed in: " + fname);
            }
        }

        std::vector<double> local(localSize);
        readDoubles(in, local.data(), local.size(), fname);

        for (int s = 0; s < nsteps; ++s) {
            for (int k = 0; k < h.n3; ++k) {
                for (int j = 0; j < h.n2; ++j) {
                    const std::size_t src =
                        static_cast<std::size_t>(h.n1) *
                        (static_cast<std::size_t>(j) +
                         static_cast<std::size_t>(h.n2) *
                             (static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(h.n3) * static_cast<std::size_t>(s)));
                    const std::size_t dst =
                        static_cast<std::size_t>(n1Global) *
                        (static_cast<std::size_t>(n2Offset + j) +
                         static_cast<std::size_t>(n2Global) *
                             (static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(n3Global) * static_cast<std::size_t>(s)));
                    std::copy(local.data() + src,
                              local.data() + src + static_cast<std::size_t>(h.n1),
                              udgf.data() + dst);
                }
            }
        }

        n2Offset += h.n2;
    }

    n1_out = n1Global;
    n2_out = n2Global;
    n3_out = n3Global;
    n4_out = nsteps;
}

void getufavg(const std::string& base,
              int nprocs,
              int npf,
              int ncu,
              std::vector<double>& uf,
              int& n1_out,
              int& n2_out,
              int& n3_out)
{
    if (nprocs <= 0) {
        throw std::runtime_error("getufavg: nprocs must be positive.");
    }
    if (npf <= 0) {
        throw std::runtime_error("getufavg: npf must be positive.");
    }
    if (ncu <= 0) {
        throw std::runtime_error("getufavg: ncu must be positive.");
    }

    struct RankData {
        int nf = 0;
        std::vector<double> values;
    };

    std::vector<RankData> ranks;
    ranks.reserve(static_cast<std::size_t>(nprocs));
    std::int64_t nfTotal = 0;

    const std::size_t block = static_cast<std::size_t>(npf) * static_cast<std::size_t>(ncu);
    for (int r = 0; r < nprocs; ++r) {
        const std::string fname = base + "_np" + std::to_string(r) + ".bin";
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            continue;
        }

        const std::int64_t bytes = fileSizeBytes(fname);
        if (bytes % static_cast<std::int64_t>(sizeof(double)) != 0) {
            throw std::runtime_error("File size not multiple of 8 in: " + fname);
        }

        const std::int64_t ndoubles =
            bytes / static_cast<std::int64_t>(sizeof(double));
        if (ndoubles < 2) {
            throw std::runtime_error("getufavg: file must contain data and nsteps: " + fname);
        }

        std::vector<double> tm(static_cast<std::size_t>(ndoubles));
        readDoubles(in, tm.data(), tm.size(), fname);

        const double nsteps = tm.back();
        if (!std::isfinite(nsteps) || nsteps <= 0.0) {
            throw std::runtime_error("getufavg: invalid nsteps in: " + fname);
        }

        const std::size_t payload = tm.size() - 1;
        if (payload % block != 0) {
            throw std::runtime_error("getufavg: payload size is incompatible with npf and ncu in: " +
                                     fname);
        }

        const int nf = static_cast<int>(payload / block);
        for (std::size_t i = 0; i < payload; ++i) {
            tm[i] /= nsteps;
        }
        tm.resize(payload);

        ranks.push_back({nf, std::move(tm)});
        nfTotal += nf;
        if (nfTotal > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("getufavg: total face count exceeds int range.");
        }
    }

    if (ranks.empty()) {
        throw std::runtime_error("getufavg: no input rank files could be opened for base: " + base);
    }

    const int nfGlobal = static_cast<int>(nfTotal);
    const std::size_t total = checkedProduct3(npf, nfGlobal, ncu, base);
    uf.assign(total, 0.0);

    int nfOffset = 0;
    for (const RankData& rank : ranks) {
        for (int c = 0; c < ncu; ++c) {
            for (int f = 0; f < rank.nf; ++f) {
                const std::size_t src =
                    static_cast<std::size_t>(npf) *
                    (static_cast<std::size_t>(f) +
                     static_cast<std::size_t>(rank.nf) * static_cast<std::size_t>(c));
                const std::size_t dst =
                    static_cast<std::size_t>(npf) *
                    (static_cast<std::size_t>(nfOffset + f) +
                     static_cast<std::size_t>(nfGlobal) * static_cast<std::size_t>(c));
                std::copy(rank.values.data() + src,
                          rank.values.data() + src + static_cast<std::size_t>(npf),
                          uf.data() + dst);
            }
        }
        nfOffset += rank.nf;
    }

    n1_out = npf;
    n2_out = nfGlobal;
    n3_out = ncu;
}

void averageudgf(const std::string& base,
                 int nprocs,
                 int nsteps,
                 int stepoffsets,
                 std::vector<double>& udgf,
                 int& n1_out,
                 int& n2_out,
                 int& n3_out)
{
    std::vector<double> snapshots;
    int n4 = 0;
    getudgf(base,
            nprocs,
            nsteps,
            stepoffsets,
            snapshots,
            n1_out,
            n2_out,
            n3_out,
            n4);

    const std::size_t slice = checkedProduct3(n1_out, n2_out, n3_out, base);
    udgf.assign(slice, 0.0);

    for (int s = 0; s < n4; ++s) {
        const double* src = snapshots.data() + slice * static_cast<std::size_t>(s);
        for (std::size_t i = 0; i < slice; ++i) {
            udgf[i] += src[i];
        }
    }

    const double scale = 1.0 / static_cast<double>(n4);
    for (double& value : udgf) {
        value *= scale;
    }
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
