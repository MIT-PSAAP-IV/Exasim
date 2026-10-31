#include "composite_config.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace exasim::composite;

struct App {
    int nsize[30] = {40, 21, 35};
    int szinteractions=0,szavfilterparam=0,szmeshadaptparam=0;
    double *avfilterparam=nullptr,*meshadaptparam=nullptr;
    double ndims[40] = {};
    int problem[35] = {};
    int flag[21] = {};
};
struct Master { int ndims[20] = {}; };

static void require(bool condition) {
    if (!condition) throw std::runtime_error("composite configuration assertion failed");
}
template<class F> static void rejects(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}

int main() {
    App app;
    require(!decode(app).enabled && decode(app).batch == 1024);
    app.nsize[0] = 20;
    require(!decode(app).enabled && decode(app).batch == 1024);
    app.nsize[0] = 40;
    app.ndims[5] = 36; app.ndims[6] = 9; app.ndims[7] = 27;
    app.ndims[9] = 14; app.ndims[10] = 9; app.ndims[11] = 3;
    app.ndims[13] = 1; app.ndims[23] = 1;
    app.problem[0] = 1; app.flag[6] = 1;
    Master master;
    master.ndims[0] = 3; master.ndims[1] = 1; master.ndims[3] = 2;
    master.ndims[5] = 27; master.ndims[6] = 9;
    for (int cap : {1, 127, 128, 256, 512, 1024, 2048, 4096, 6928}) {
        app.ndims[24] = cap;
        require(decode(app).enabled && decode(app).batch == cap);
        validateShape<double>(decode(app), app, master);
    }
    for (double invalid : {-1., 0.5, 2., double(INFINITY), double(NAN)}) {
        app.ndims[23] = invalid;
        rejects([&] { decode(app); });
    }
    app.ndims[23] = 1;
    for (double invalid : {-1., 1024.5, 2147483648., double(INFINITY), double(NAN)}) {
        app.ndims[24] = invalid;
        rejects([&] { decode(app); });
    }
    app.ndims[24] = 0; // Old zero-filled reserved slots retain the default.
    require(decode(app).batch == 1024);
    for (int cap : {6929, 8192, 2147483647}) {
        app.ndims[24] = cap;
        rejects([&] { validateShape<double>(decode(app), app, master); });
    }
    app.ndims[24] = 1024;
    rejects([&] { validateShape<float>(decode(app), app, master); });
    app.problem[0] = 0;
    rejects([&] { validateShape<double>(decode(app), app, master); });
    app.problem[0] = 1;
    app.problem[34] = 1;
    rejects([&] { validateShape<double>(decode(app), app, master); });
    app.problem[34] = 0;
    app.nsize[24] = 10; app.szinteractions = 10;
    rejects([&] { validateShape<double>(decode(app), app, master); });
    app.nsize[24] = 0; app.szinteractions = 0;
    app.problem[28] = 1;
    rejects([&] { validateShape<double>(decode(app), app, master); });
    app.problem[28] = 0;
    // Optional blocks are absent in older app.bin headers. Use an actual
    // 16-entry allocation to detect any read of the absent nsize[24] slot.
    struct ShortApp : App {
        std::unique_ptr<int[]> storage;
        int* nsize;
        explicit ShortApp(const App& source):App(source),storage(new int[16]),nsize(storage.get()) {
            std::copy_n(App::nsize,16,nsize);
        }
    } shortApp(app);
    validateShape<double>(decode(shortApp),shortApp,master);
    double avfilter[2]={0.,1.},meshadapt[20]={};
    app.szavfilterparam=2;app.avfilterparam=avfilter;
    app.szmeshadaptparam=20;app.meshadaptparam=meshadapt;
    validateOptions(decode(app),app);
    validateShape<double>(decode(app),app,master); // Current method0/noadapt stays supported.
    avfilter[0]=1.;
    rejects([&] { validateOptions(decode(app),app); });
    rejects([&] { validateShape<double>(decode(app),app,master); });
    avfilter[0]=0.;meshadapt[0]=1.;
    rejects([&] { validateOptions(decode(app),app); });
    rejects([&] { validateShape<double>(decode(app),app,master); });
    for(double invalid : {double(NAN),double(INFINITY),0.5}) {
        avfilter[0]=invalid;meshadapt[0]=0.;
        rejects([&] { validateOptions(decode(app),app); });
        avfilter[0]=0.;meshadapt[0]=invalid;
        rejects([&] { validateOptions(decode(app),app); });
    }
    app.ndims[23]=0;avfilter[0]=1.;meshadapt[0]=1.;
    validateOptions(decode(app),app); // Feature off retains existing auxiliary options.
    app.ndims[23]=1;avfilter[0]=meshadapt[0]=0.;
    master.ndims[0] = 2;
    rejects([&] { validateShape<double>(decode(app), app, master); });
    // A real P2 quadrilateral has eight physical fields, not a truncated 3D
    // declaration. Reject each mismatched geometry/component field independently.
    app.ndims[5]=24; app.ndims[6]=8; app.ndims[7]=16;
    app.ndims[9]=13; app.ndims[10]=8; app.ndims[11]=2;
    master.ndims[5]=9; master.ndims[6]=3;
    validateShape<double>(decode(app),app,master);
    for(int slot : {5,6,7,9,10,11,13}) {
        ++app.ndims[slot];
        rejects([&] { validateShape<double>(decode(app),app,master); });
        --app.ndims[slot];
    }
    for(int slot : {0,1,3,5,6}) {
        --master.ndims[slot];
        rejects([&] { validateShape<double>(decode(app),app,master); });
        ++master.ndims[slot];
    }
    app.ndims[23] = 0;
    validateShape<float>(decode(app), app, master); // No shape restriction when off.
    require(count(2147483647, "edge") == 2147483647);
    rejects([] { count(2147483648ULL, "edge"); });
    checkTemporaryCounts(1024, 13257, 2048, 0, 64, 16);
    rejects([] { checkTemporaryCounts(-1, 1, 1, 0, 64, 16); });
    rejects([] { checkTemporaryCounts(1024, 3000000, 2048, 0, 64, 16); });
    for (const char* text : {"0", "-1", "1024.5", "NaN", "Inf", "-Inf",
                             "2147483648", "1e1000", "1024oops", "[1024]", "\"1024\"", ""})
        rejects([&] { parseInteger(text, "compositebatchsize", 1, 2147483647); });
    require(parseInteger(" 1.024e3 ", "compositebatchsize", 1, 2147483647) == 1024);
    std::cout << "COMPOSITE_CONFIG_PASS\n";
}
