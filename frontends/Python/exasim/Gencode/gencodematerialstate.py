import os

from .varsassign import varsassign
from .sympyassign import sympyassign
from .sympyassign2 import sympyassign2


def gencodematerialstate(filename, f, xdg, udg, odg, wdg, uinf, param, time, foldername):
    cpufile = "Kokkos" + filename
    tmp = "(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    str_code = "\tKokkos::parallel_for(" + '"' + filename + '"' + ", ng, KOKKOS_LAMBDA(const size_t i) {\n"
    strkk = "void " + cpufile + tmp + "{\n" + str_code

    fstr = str(f.flatten('F'))
    str_code = ""
    str_code = varsassign(str_code, "param", len(param), 0, fstr)
    str_code = varsassign(str_code, "uinf", len(uinf), 0, fstr)
    str_code = varsassign(str_code, "xdg", len(xdg), 1, fstr)
    str_code = varsassign(str_code, "udg", len(udg), 1, fstr)
    str_code = varsassign(str_code, "odg", len(odg), 1, fstr)
    str_code = varsassign(str_code, "wdg", len(wdg), 1, fstr)
    str_code = sympyassign(str_code, f)

    strkk = strkk + str_code + "\t});\n" + "}\n\n"
    strkk = strkk.replace("T ", "dstype ")

    with open(os.path.join(foldername, cpufile + ".cpp"), "w") as fid:
        fid.write(strkk)

    return 0


def nocodematerialstate(filename, foldername):
    cpufile = "Kokkos" + filename
    tmp = "(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    strkk = "void " + cpufile + tmp + "{\n"
    strkk += "    (void)f; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)uinf;\n"
    strkk += "    (void)param; (void)time; (void)modelnumber; (void)ng; (void)nc;\n"
    strkk += "    (void)ncu; (void)nd; (void)ncx; (void)nco; (void)ncw; (void)nmaterialstate;\n"
    strkk += "}\n"

    with open(os.path.join(foldername, cpufile + ".cpp"), "w") as fid:
        fid.write(strkk)

    return 0


def hdggencodematerialstate(filename, f, xdg, udg, odg, wdg, uinf, param, time, foldername):
    cpufile = "Hdg" + filename
    tmp = "(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    str_code = "\tKokkos::parallel_for(" + '"' + filename + '"' + ", ng, KOKKOS_LAMBDA(const size_t i) {\n"
    strkk = "void " + cpufile + tmp + "{\n" + str_code

    fstr = str(f.flatten('F'))
    str_code = ""
    str_code = varsassign(str_code, "param", len(param), 0, fstr)
    str_code = varsassign(str_code, "uinf", len(uinf), 0, fstr)
    str_code = varsassign(str_code, "xdg", len(xdg), 1, fstr)
    str_code = varsassign(str_code, "udg", len(udg), 1, fstr)
    str_code = varsassign(str_code, "odg", len(odg), 1, fstr)
    str_code = varsassign(str_code, "wdg", len(wdg), 1, fstr)
    str_code = sympyassign2(str_code, f, udg, wdg, None)

    strkk = strkk + str_code + "\t});\n" + "}\n\n"
    strkk = strkk.replace("T ", "dstype ")

    with open(os.path.join(foldername, cpufile + ".cpp"), "w") as fid:
        fid.write(strkk)

    return 0


def hdgnocodematerialstate(filename, foldername):
    cpufile = "Hdg" + filename
    tmp = "(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nmaterialstate)\n"
    strkk = "void " + cpufile + tmp + "{\n"
    strkk += "    (void)f; (void)f_udg; (void)f_wdg; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)uinf;\n"
    strkk += "    (void)param; (void)time; (void)modelnumber; (void)ng; (void)nc;\n"
    strkk += "    (void)ncu; (void)nd; (void)ncx; (void)nco; (void)ncw; (void)nmaterialstate;\n"
    strkk += "}\n"

    with open(os.path.join(foldername, cpufile + ".cpp"), "w") as fid:
        fid.write(strkk)

    return 0
