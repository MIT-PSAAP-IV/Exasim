import copy as pycopy
import itertools
import os
import numpy
from numpy import *
from .. import Preprocessing
from .. import Gencode
from .. import config
from .fetchsolution import fetchsolution
from .getsolutions import getsolutions
from .generatecode import generatecode

def exasim(pde,mesh):

    if isinstance(pde, list):
        nmodels = len(pde);      
    else:
        nmodels = 1
    
    res = None;
    if nmodels == 1:
        # search compilers and set options
        pde = Gencode.setcompilers(pde);
        physicsparamcases, has_physicsparam_sweep = _normalize_physicsparam_sweep(pde)
        if has_physicsparam_sweep:
            pde['physicsparam'] = physicsparamcases[0].copy()

        # generate input files and store them in datain folder
        pde, mesh, master, dmd = Preprocessing.preprocessing(pde,mesh);
        writeappbase = _writeapp_template(pde) if has_physicsparam_sweep else None

        # generate source codes and store them in app folder
        Gencode.gencode(pde);

        # compile source codes to build an executable file and store it in build folder
        compilerstr = Gencode.cmakecompile(pde);
        # compilerstr = Gencode.compilepdemodel(pde);

        # When pde['exportapp'] is set, package a relocatable "data transfer
        # app" bundle INSTEAD of running the simulation here: a production run
        # can be prohibitively expensive on the local machine, so we export it
        # to be built and run elsewhere. pde['buildandrun'] (default True)
        # controls whether exportapp builds+runs the bundle in a scratch dir to
        # verify it; set it False to export without any local build/run.
        pde['vistime'] = [];
        if pde.get('exportapp'):
            Gencode.exportapp(pde, pde['exportapp'], build=pde.get('buildandrun', True));
            runstr = None;
            sol = None;
        else:
            if has_physicsparam_sweep:
                sol = []
                runstr = []
                res = []
                pde['paramcaseoutputdirs'] = []
                warmstart = pde.get('physicsparamwarmstart', 0) == 1
                previousoutdir = None
                for icase, physicsparam in enumerate(physicsparamcases, start=1):
                    pdecase = pycopy.deepcopy(writeappbase)
                    pdecase['physicsparam'] = physicsparam.copy()
                    pdecase['dataoutpath'] = _paramcase_output_dir(pdecase, icase)
                    os.makedirs(pdecase['dataoutpath'], exist_ok=True)
                    _write_physicsparam_case(pdecase['dataoutpath'], pdecase['physicsparam'])

                    if warmstart and icase > 1:
                        _write_physicsparam_warmstart(pdecase, dmd, previousoutdir)
                    pdecase = Preprocessing.writeapp(pdecase, _app_bin_file(pdecase))
                    runstr.append(Gencode.runcode(pdecase, 1))
                    sol.append(fetchsolution(pdecase, master, dmd, pdecase['dataoutpath']))

                    if pdecase['saveResNorm']:
                        fn = os.path.join(pdecase['dataoutpath'], "out_residualnorms0.bin")
                        tm = fromfile(open(fn, "r"), dtype=float64)
                        ne = int(round(size(tm)/(4)))
                        tm = reshape(tm,[4,ne],'F')
                        res.append(tm.transpose())
                    else:
                        res.append(None)

                    pde['paramcaseoutputdirs'].append(pdecase['dataoutpath'])
                    previousoutdir = pdecase['dataoutpath']
                _write_physicsparam_sweep_manifest(
                    _dataout_base_dir(pde), physicsparamcases, pde['paramcaseoutputdirs'], warmstart)
            else:
                runstr = Gencode.runcode(pde, 1);

                # get solution from output files in dataout folder (per-model strn
                # dir; strn="" for model 0 keeps the historical datapath/dataout).
                strn = config.model_strn(pde)
                dataout_dir = os.path.join(pde['datapath'], "dataout", strn) if strn \
                    else (pde['datapath'] + "/dataout")
                sol = fetchsolution(pde,master,dmd, dataout_dir);
                #sol, _, _ = getsolutions(pde, dmd);

                if pde['saveResNorm']:
                    fn = "dataout/out_residualnorms0.bin";
                    tm = fromfile(open(fn, "r"), dtype=float64);
                    ne = int(round(size(tm)/(4)));
                    tm = reshape(tm,[4,ne],'F');
                    res = tm.transpose();

    else:        
        master = [None] * nmodels
        dmd = [None] * nmodels
        sol = [None] * nmodels
        res = [None] * nmodels

        # Combined multi-PDE: model m occupies slot m. Assigning modelnumber=m
        # gives each model a distinct modelid (100+m), a distinct kernel/build
        # dir (config.model_builddir) and distinct datain/dataout subdirs, so
        # the N models generate without clobbering and link into one exasimapp.
        for m in range(0, nmodels):
            pde[m]['modelnumber'] = m

        # preprocess and generate code (own kernel dir) for all PDE models
        for m in range(0, nmodels):
            pde[m],mesh[m],master[m],dmd[m] = generatecode(pde[m],mesh[m])[0:4];

        # build ONE executable that dispatches all model ids, then run it with
        # one (datain, dataout) pair per model (the solver's multi-model CLI).
        compilerstr = Gencode.cmakecompile_combined(pde);

        # Export the combined bundle INSTEAD of running locally when requested
        # (see the single-model branch). pde[0]['buildandrun'] (default True)
        # gates the scratch build+run verification.
        if pde[0].get('exportapp'):
            Gencode.exportapp_combined(pde, pde[0]['exportapp'], build=pde[0].get('buildandrun', True));
            runstr = None;
        else:
            runstr = Gencode.runcode_combined(pde);

            # get solution from each model's own dataout subdir
            for m in range(0, nmodels):
                strn = config.model_strn(pde[m])
                doutdir = os.path.join(pde[m]['datapath'], "dataout", strn) if strn \
                    else (pde[m]['datapath'] + "/dataout")
                sol[m] = fetchsolution(pde[m],master[m],dmd[m], doutdir);

                if pde[m]['saveResNorm']:
                    fn = os.path.join(doutdir, "out_residualnorms0.bin");
                    tm = fromfile(open(fn, "r"), dtype=float64);
                    ne = int(round(size(tm)/(4)));
                    tm = reshape(tm,[4,ne],'F');
                    res[m] = tm.transpose();

    return sol,pde,mesh,master,dmd,compilerstr,runstr,res


def _normalize_physicsparam_sweep(pde):
    base = numpy.asarray(pde['physicsparam'], dtype=float).reshape(-1)
    spec = pde.get('physicsparamsweep')
    if not _empty_sweep_spec(spec):
        cases = _physicsparam_sweep_cases(spec, base.size)
        return cases, cases.shape[0] > 1

    raw = numpy.asarray(pde['physicsparam'], dtype=float)
    if raw.ndim == 2 and raw.shape[0] > 1 and raw.shape[1] > 1:
        _validate_physicsparam_cases(raw, raw.shape[1])
        return raw, True

    return base.reshape(1, -1), False


def _empty_sweep_spec(spec):
    if spec is None:
        return True
    if isinstance(spec, dict):
        return len(spec) == 0
    try:
        return len(spec) == 0
    except TypeError:
        return False


def _physicsparam_sweep_cases(spec, nparam):
    if isinstance(spec, dict):
        if 'samples' in spec:
            return _physicsparam_sweep_cases(spec['samples'], nparam)
        if 'values' in spec:
            return _physicsparam_sweep_cases(spec['values'], nparam)
        if 'grid' in spec:
            return _physicsparam_grid_cases(spec['grid'], nparam)
        raise ValueError("physicsparamsweep dict must contain 'samples', 'values', or 'grid'.")

    if isinstance(spec, (list, tuple)) and spec and not numpy.isscalar(spec[0]):
        rows = [numpy.asarray(v, dtype=float).reshape(-1) for v in spec]
        for i, row in enumerate(rows, start=1):
            if row.size != nparam:
                raise ValueError(f"physicsparamsweep case {i} has {row.size} parameters; expected {nparam}.")
        cases = numpy.vstack(rows)
    else:
        cases = numpy.asarray(spec, dtype=float)
        if cases.ndim == 1:
            cases = cases.reshape(-1, 1) if nparam == 1 and cases.size > 1 else cases.reshape(1, -1)

    _validate_physicsparam_cases(cases, nparam)
    return cases


def _physicsparam_grid_cases(grid, nparam):
    if len(grid) != nparam:
        raise ValueError("physicsparamsweep['grid'] must contain one value list per physics parameter.")
    rows = list(itertools.product(*grid))
    cases = numpy.asarray(rows, dtype=float)
    _validate_physicsparam_cases(cases, nparam)
    return cases


def _validate_physicsparam_cases(cases, nparam):
    if cases.ndim != 2 or cases.shape[1] != nparam:
        raise ValueError(f"Each physicsparamsweep row must contain {nparam} physics parameters.")
    if not numpy.all(numpy.isfinite(cases)):
        raise ValueError("physicsparamsweep cases must contain finite numeric values.")


def _paramcase_output_dir(pde, icase):
    return os.path.join(_dataout_base_dir(pde), f"paramcase_{icase:04d}")


def _dataout_base_dir(pde):
    strn = config.model_strn(pde)
    return os.path.join(pde['datapath'], "dataout", strn) if strn else os.path.join(pde['datapath'], "dataout")


def _write_physicsparam_case(outdir, values):
    numpy.savetxt(os.path.join(outdir, "physicsparam.txt"),
                  numpy.asarray(values, dtype=float).reshape(1, -1),
                  fmt="%.17g")


def _write_physicsparam_warmstart(pde, dmd, previousoutdir):
    udg = getsolutions(os.path.join(previousoutdir, "outudg"), dmd)[:, :, :, -1]
    wdg = getsolutions(os.path.join(previousoutdir, "outwdg"), dmd) \
        if pde.get('ncw', 0) > 0 else None
    if wdg is not None:
        wdg = wdg[:, :, :, -1]

    strn = config.model_strn(pde)
    datain = os.path.join(pde['datapath'], "datain", strn)
    for iproc, part in enumerate(dmd, start=1):
        solfile = os.path.join(datain, "sol.bin" if pde['mpiprocs'] == 1
                               else f"sol{iproc}.bin")
        elempart = numpy.asarray(part['elempart']).reshape(-1)
        if elempart.size > 0 and elempart.min() == 1:
            elempart = elempart - 1
        _rewrite_warmstart_sol_file(solfile, udg[:, :, elempart],
                                    None if wdg is None else wdg[:, :, elempart])


def _rewrite_warmstart_sol_file(solfile, udgpart, wdgpart):
    data = numpy.fromfile(solfile, dtype=numpy.float64)
    if data.size == 0:
        raise RuntimeError(f"Warm-start solution file {solfile} is empty.")

    nsize_len = int(data[0])
    nsize_start = 1
    nsize_end = nsize_start + nsize_len
    nsize = data[nsize_start:nsize_end].astype(numpy.int64)
    ndims_start = nsize_end
    ndims_end = ndims_start + int(nsize[0])
    xdg_start = ndims_end
    xdg_end = xdg_start + int(nsize[1])
    udg_start = xdg_end
    udg_end = udg_start + int(nsize[2])
    odg_start = udg_end
    odg_end = odg_start + int(nsize[3])
    wdg_start = odg_end
    wdg_end = wdg_start + int(nsize[4])

    udgflat = numpy.asarray(udgpart, dtype=numpy.float64).reshape(-1, order='F')
    newnsize = nsize.copy()
    newnsize[2] = udgflat.size
    wdgflat = data[wdg_start:wdg_end]

    if wdgpart is not None:
        wdgflat = numpy.asarray(wdgpart, dtype=numpy.float64).reshape(-1, order='F')
        newnsize[4] = wdgflat.size

    data = numpy.concatenate((
        data[:nsize_start],
        newnsize.astype(numpy.float64),
        data[ndims_start:xdg_end],
        udgflat,
        data[odg_start:odg_end],
        wdgflat,
        data[wdg_end:],
    ))

    data.tofile(solfile)


def _write_physicsparam_sweep_manifest(baseout, cases, outdirs, warmstart):
    with open(os.path.join(baseout, "physicsparam_sweep_manifest.txt"), "w") as f:
        f.write(f"ncases {cases.shape[0]}\n")
        f.write(f"nparam {cases.shape[1]}\n")
        f.write(f"physicsparamwarmstart {1 if warmstart else 0}\n")
        for i, row in enumerate(cases, start=1):
            values = " ".join(f"{float(v):.17g}" for v in row)
            f.write(f"case {i} {outdirs[i - 1]} {values}\n")


def _writeapp_template(app):
    template = pycopy.deepcopy(app)
    template['flag'] = numpy.asarray(template['flag']).reshape(-1, order='F')[19:]
    template['problem'] = numpy.asarray(template['problem']).reshape(-1, order='F')[28:]
    template['factor'] = numpy.asarray(template['factor']).reshape(-1, order='F')[5:]
    template['solversparam'] = numpy.asarray(template['solversparam']).reshape(-1, order='F')[4:]
    return template


def _app_bin_file(pde):
    strn = config.model_strn(pde)
    return os.path.join(pde['datapath'], "datain", strn, "app.bin")
