function gencodematerialstate(filename::String, f, xdg, udg, odg, wdg, uinf, param, time, foldername)

    cpufile = "Kokkos" * filename
    tmp = "(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    str = "\tKokkos::parallel_for(" * "\"" * filename  * "\"" * ", ng, KOKKOS_LAMBDA(const size_t i) {\n"
    strkk = "void " * cpufile * tmp * "{\n" * str

    fstr = string(f[:])
    str = ""
    str = varsassign(str, "param", length(param), 0, fstr)
    str = varsassign(str, "uinf", length(uinf), 0, fstr)
    str = varsassign(str, "xdg", length(xdg), 1, fstr)
    str = varsassign(str, "udg", length(udg), 1, fstr)
    str = varsassign(str, "odg", length(odg), 1, fstr)
    str = varsassign(str, "wdg", length(wdg), 1, fstr)
    str = sympyassign(str, f)

    strkk = strkk * str * "\t});\n" * "}\n\n"
    strkk = replace(strkk, "T " => "dstype ")

    open(joinpath(foldername, cpufile * ".cpp"), "w") do fid
        write(fid, strkk)
    end

    return 0
end

function nocodematerialstate(filename::String, foldername::String)

    cpufile = "Kokkos" * filename
    tmp = "(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    strkk = "void " * cpufile * tmp * "{\n"
    strkk *= "    (void)f; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)uinf;\n"
    strkk *= "    (void)param; (void)time; (void)modelnumber; (void)ng; (void)nc;\n"
    strkk *= "    (void)ncu; (void)nd; (void)ncx; (void)nco; (void)ncw; (void)nmaterialstate;\n"
    strkk *= "}\n"

    open(joinpath(foldername, cpufile * ".cpp"), "w") do fid
        write(fid, strkk)
    end

    return 0
end

function hdggencodematerialstate(filename::String, f, xdg, udg, odg, wdg, uinf, param, time, foldername)

    cpufile = "Hdg" * filename
    tmp = "(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    str = "\tKokkos::parallel_for(" * "\"" * filename  * "\"" * ", ng, KOKKOS_LAMBDA(const size_t i) {\n"
    strkk = "void " * cpufile * tmp * "{\n" * str

    fstr = string(f[:])
    str = ""
    str = varsassign(str, "param", length(param), 0, fstr)
    str = varsassign(str, "uinf", length(uinf), 0, fstr)
    str = varsassign(str, "xdg", length(xdg), 1, fstr)
    str = varsassign(str, "udg", length(udg), 1, fstr)
    str = varsassign(str, "odg", length(odg), 1, fstr)
    str = varsassign(str, "wdg", length(wdg), 1, fstr)
    str = sympyassign2(str, f[:], udg, wdg, nothing)

    strkk = strkk * str * "\t});\n" * "}\n\n"
    strkk = replace(strkk, "T " => "dstype ")

    open(joinpath(foldername, cpufile * ".cpp"), "w") do fid
        write(fid, strkk)
    end

    return 0
end

function hdgnocodematerialstate(filename::String, foldername::String)

    cpufile = "Hdg" * filename
    tmp = "(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    strkk = "void " * cpufile * tmp * "{\n"
    strkk *= "    (void)f; (void)f_udg; (void)f_wdg; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)uinf;\n"
    strkk *= "    (void)param; (void)time; (void)modelnumber; (void)ng; (void)nc;\n"
    strkk *= "    (void)ncu; (void)nd; (void)ncx; (void)nco; (void)ncw; (void)nmaterialstate;\n"
    strkk *= "}\n"

    open(joinpath(foldername, cpufile * ".cpp"), "w") do fid
        write(fid, strkk)
    end

    return 0
end
