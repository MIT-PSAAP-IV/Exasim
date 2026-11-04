// Uses the actual text reader and binary writer, without solver dependencies.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
#ifndef EXASIM_INSTALL_PREFIX
#define EXASIM_INSTALL_PREFIX ""
#endif
using idx_t = int; // Only the unused DMD declaration needs the partition index type.
#ifdef EXASIM_COMPOSITE_TEST_TEXT2CODE
// Exactly the production Text2Code parser/helper implementation, not the backend copy.
#include "text2code/text2code/TextParser.hpp"
#include "text2code/text2code/tinyexpr.cpp"
#include "text2code/text2code/helpers.cpp"
#include "text2code/text2code/readpdeapp.cpp"
#else
#include "structs.hpp"
#include "TextParser.hpp"
#include "tinyexpr.cpp"
#include "helpers.cpp"
#ifdef EXASIM_COMPOSITE_TEST_HPP
#include "readpdeapp.hpp"
#else
#include "readpdeapp.cpp"
#endif
#endif

static void require(bool condition, const string& message) {
    if (!condition) throw runtime_error(message);
}
static PDE convert(const filesystem::path& input, const filesystem::path& output) {
    auto params = parseInputFile(input.string(), 1);
    auto pde = initializePDE(params, 1);
    writepde(pde, output.string());
    return pde;
}
static vector<double> dimensions(const filesystem::path& input) {
    ifstream file(input, ios::binary);
    double count = 0.; file.read(reinterpret_cast<char*>(&count), sizeof(double));
    require(count == 30., "unexpected binary header");
    vector<double> sizes(30), result(40);
    file.read(reinterpret_cast<char*>(sizes.data()), sizes.size()*sizeof(double));
    require(sizes[0] >= 40., "unexpected ndims length");
    result.resize(static_cast<size_t>(sizes[0]));
    file.read(reinterpret_cast<char*>(result.data()), result.size()*sizeof(double));
    require(bool(file), "incomplete app.bin");
    return result;
}
int main(int argc, char** argv) {
    if (argc == 4 && string(argv[1]) == "--convert") {
        convert(argv[2], argv[3]);
        return 0;
    }
    const auto root = filesystem::temp_directory_path() /
        ("exasim-composite-config-" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::create_directory(root);
    struct Cleanup { filesystem::path path; ~Cleanup() { filesystem::remove_all(path); } } cleanup{root};
    const auto input = root / "pdeapp.txt", output = root / "app.bin";
    auto write = [&](const string& content) {
        ofstream file(input);
        file << "model = \"ModelD\";\nmodelfile = \"unused.txt\";\nmeshfile = \"unused.bin\";\n"
             << "platform = \"cpu\";\nmpiprocs = 1;\nporder = 2;\npgauss = 4;\n"
             << "discretization = \"hdg\";\n"
             << "physicsparam = [1];\ntau = [1];\nboundaryconditions = [1];\n"
             << "boundaryexpressions = {\"x1 < 0\"};\n" << content;
    };
    write("discretization = \"hdg\";\ncompositehdg = 1;\ncompositebatchsize = 1.024e3;\nuniformrefinementlevel = 2;\n");
    auto pde = convert(input, output);
    auto dims = dimensions(output);
    require(pde.compositehdg == 1 && pde.compositebatchsize == 1024, "text decode mismatch");
    require(dims[23] == 1 && dims[24] == 1024 && dims[22] == 0, "binary slot mismatch");
    require(pde.problem.size() > 33 && pde.problem[33] == 2, "upstream refinement slot changed");
    write("compositehdg = 1;\ncompositebatchsize = 127;\n");
    pde = convert(input, output); dims = dimensions(output);
    require(dims[23] == 1 && dims[24] == 127, "explicit composite cap lost in app.bin");
    for (const string& value : {"5", "[1001, 5, 5]", "[5 1001]", "[]"}) {
        write("compositehdg = 1;\ncompositebc = " + value + ";\n");
        pde = convert(input,output); dims = dimensions(output);
        auto expected = exasim::composite::parseBoundaryCodes(value);
        require(pde.compositebc == expected,"boundary text mismatch");
        require(exasim::composite::decodeBoundaryCodes(dims.data(),dims.size()) == expected,"boundary binary mismatch");
    }
    for (const string& value : {"0", "-1", "[5.5]", "[NaN]", "[Inf]", "[2147483648]", "[5,,2]", "[5,]", "[5;2]", "[5]junk"}) {
        write("compositebc = " + value + ";\n");
        bool rejected = false;
        try { convert(input,output); } catch (const exception&) { rejected = true; }
        require(rejected,"accepted invalid compositebc: " + value);
    }
    write("compositehdg = 1;\ninterfacetype = 1;\n");
    // This release rejects nonmatching input with error()/exit, not an exception.
    // Isolate that production guard and require its specific diagnostic as well
    // as failure, so a launch error cannot masquerade as a successful check.
    const auto diagnostic = root / "nonmatching.log";
    ostringstream command;
    command << quoted(filesystem::absolute(argv[0]).string()) << " --convert "
            << quoted(input.string()) << ' ' << quoted(output.string())
            << " > " << quoted(diagnostic.string()) << " 2>&1";
    const int status = std::system(command.str().c_str());
    ifstream log(diagnostic);
    const string message((istreambuf_iterator<char>(log)), istreambuf_iterator<char>());
    require(status != 0 && message.find("Nonmatching interfaces and generalized interactions are not available in this source release") != string::npos,
            "unsupported composite nonmatching text was not rejected with its diagnostic");
    for (const string& key : {"compositehdg", "compositebatchsize"}) {
        vector<string> invalid = {"-1", "0.5", "NaN", "Inf", "-Inf", "2147483648", "1e1000", "1junk", "[1]", "\"1\""};
        invalid.push_back(key == "compositehdg" ? "2" : "0");
        for (const auto& value : invalid) {
            write(key + " = " + value + ";\n");
            bool rejected = false;
            try { convert(input, output); } catch (const exception&) { rejected = true; }
            require(rejected, "text reader accepted " + key + "=" + value);
        }
    }
    write("discretization = \"hdg\";\n");
    pde = convert(input, output); dims = dimensions(output);
    require(pde.compositehdg == 0 && pde.compositebatchsize == 1024, "legacy text defaults changed");
    require(dims[23] == 0 && dims[24] == 1024, "default binary slots changed");
    pde.compositebatchsize = 0;
    bool rejected = false;
    try { writepde(pde, output.string()); } catch (const exception&) { rejected = true; }
    require(rejected, "programmatic invalid cap accepted");
    std::cout << "COMPOSITE_TEXT_BINARY_PASS\n";
}
