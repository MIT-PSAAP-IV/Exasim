from numpy import *

def writeapp(app,filename):
    def flat(key):
        if key not in app:
            app[key] = array([]);
        return array(app[key]).flatten(order='F')

    app['flag'] = array(app['flag']);
    app['problem'] = array(app['problem']);
    app['factor'] = array(app['factor']);
    app['solversparam'] = array(app['solversparam']);
    if 'physicsparamwarmstart' not in app:
        app['physicsparamwarmstart'] = 0;
    if 'builtinmodelID' not in app:
        app['builtinmodelID'] = 0;
    if 'frontendgenerated' not in app:
        app['frontendgenerated'] = 1;
    if 'uniformrefinementlevel' not in app:
        app['uniformrefinementlevel'] = 0;

    appname = 0;
    tmp = array([app['tdep'], app['wave'], app['linearproblem'], app['debugmode'], app['matvecorder'], app['GMRESortho'], app['preconditioner'], app['precMatrixType'], app['NLMatrixType'], app['runmode'], app['tdfunc'], app['source'], app['modelnumber'], app['extFhat'], app['extUhat'], app['extStab'], app['subproblem'], app['saveParaview'], app['physicsparamwarmstart'], app['builtinmodelID'], app['frontendgenerated']]);
    app['flag'] =  concatenate([tmp,app['flag']]);
    # problem[0..27], coupling slots problem[28..31], STG chemistry at problem[32],
    # then uniform refinement at problem[33].
    # Keep these fixed slots aligned with the Matlab/Julia and C++ preprocessors.
    # Defensive .get keeps older caller-created dictionaries working.
    tmp = array([app['hybrid'], appname, app['temporalscheme'], app['torder'], app['nstage'], app['convStabMethod'], app['diffStabMethod'], app['rotatingFrame'], app['viscosityModel'], app['SGSmodel'], app['ALE'], app['AV'], app['linearsolver'], app['NLiter'], app['linearsolveriter'], app['GMRESrestart'], app['RBdim'], app['saveSolFreq'], app['saveSolOpt'], app['timestepOffset'], app['stgNmode'], app['saveSolBouFreq'], app['ibs'], app['dae_steps'], app['saveResNorm'], app['AVsmoothingIter'], app['frozenAVflag'], app['ppdegree'], app.get('coupledinterface', 0), app.get('coupledcondition', 0), app.get('coupledboundarycondition', 0), app.get('AVdistfunction', 0), app.get('stgchem', 0), app['uniformrefinementlevel']]);
    app['problem'] = concatenate([tmp, app['problem']]);
    tmp = array([app['time'], app['dae_alpha'], app['dae_beta'], app['dae_gamma'], app['dae_epsilon']])    
    app['factor'] = concatenate([tmp, app['factor']]);
    tmp = array([app['NLtol'], app['linearsolvertol'], app['matvectol'], app['NLparam']]);
    app['solversparam'] = concatenate([tmp, app['solversparam']]);

    app['flag'] = array(app['flag']).flatten('F');
    app['problem'] = array(app['problem']);
    app['factor'] = array(app['factor']);
    app['solversparam'] = array(app['solversparam']);

    ndims = zeros((40,1));
    ndims[1-1] = app['mpiprocs'];  # number of processors
    ndims[2-1] = app['nd'];
    ndims[3-1] = 0;
    ndims[4-1] = 0;
    ndims[5-1] = 0;
    ndims[6-1] = app['nc'];
    ndims[7-1] = app['ncu'];
    ndims[8-1] = app['ncq'];
    ndims[9-1] = app['ncp'];
    ndims[10-1] = app['nco'];
    ndims[11-1] = app['nch'];
    ndims[12-1] = app['ncx'];
    ndims[13-1] = app['nce'];
    ndims[14-1] = app['ncw'];
    ndims[15-1] = app['nsca'];
    ndims[16-1] = app['nvec'];
    ndims[17-1] = app['nten'];
    ndims[18-1] = app['nbqoi'];
    ndims[19-1] = app['nvqoi'];

    #if app['nco'] != app['vindx'].shape[0]:  #size(app.vindx,1):
    #    error("app.nco mus be equal to size(app.vindx,1)");

    aviter = int(app.get('AVcontinuationIter', 0))
    if aviter >= 2:
        alpha = float(app.get('AVcontinuationLogScale', 1.0))
        coeff_start = float(app.get('AVcoeffStart', 0.0))
        coeff_end = float(app.get('AVcoeffEnd', 0.0))
        if not all(isfinite([alpha, coeff_start, coeff_end])):
            raise ValueError("AV continuation parameters must be finite.")
        t = linspace(0.0, 1.0, aviter)
        if abs(alpha) <= 1.0e-14:
            g1 = 1.0 - t
            g2 = t
        else:
            denominator = expm1(alpha)
            g1 = expm1(alpha * (1.0 - t)) / denominator
            g2 = expm1(alpha * t) / denominator
        app['avparam1'] = coeff_start * g1
        app['avparam2'] = coeff_end * g2
        app['avparam1'][[0, -1]] = [coeff_start, 0.0]
        app['avparam2'][[0, -1]] = [0.0, coeff_end]
    avparam1 = flat('avparam1')
    avparam2 = flat('avparam2')
    if size(avparam1) != size(avparam2):
        raise ValueError("avparam1 and avparam2 must have the same length.")
    avparam = empty(2 * size(avparam1), dtype=float64)
    avparam[0::2] = avparam1
    avparam[1::2] = avparam2
    avfilterparam = array([app.get('AVsmoothingMethod', 0),
                           app.get('AVHelmholtzCoeff', 1.0)], dtype=float64)
    meshadaptparam = array([
        app.get('meshadaptenabled', 0), app.get('meshadaptfield', 1),
        app.get('meshadaptavcomponent', 1), app.get('meshadaptsmoothingpasses', 30),
        app.get('meshadaptiterations', 1), app.get('meshadaptalpha', 0.25),
        app.get('meshadaptqmin', 0.2), app.get('meshadaptqmax', 0.8),
        app.get('meshadaptHelmholtzCoeff', 0.02), app.get('meshadapttargetexponent', 2.0),
        app.get('meshadaptpoissonratio', 0.2), app.get('meshadaptyoungmodulus', 1.0),
        app.get('meshadaptminimumyoungmodulus', 1.0e-3), app.get('meshadaptshearscale', 1.0),
        app.get('meshadaptvolumetricscale', 1.0), app.get('meshadaptforcescale', 1.0),
        app.get('meshadaptdamping', 1.0), app.get('meshadaptminimumjacobianratio', 1.0e-8),
        app.get('meshadaptHelmholtzTau', 2.0), app.get('meshadaptelasticitytau', 1.0e3)
    ], dtype=float64)
    meshadaptbcs = array(app.get('meshadaptboundaryconditions', []), dtype=float64).flatten(order='F')
    distanceboundaryconditions = array(app.get('distanceboundaryconditions', []), dtype=float64).flatten(order='F')

    nsize = zeros((30,1));
    nsize[1-1] = size(ndims);
    nsize[2-1] = size(app['flag']);  # size of flag
    nsize[3-1] = size(app['problem']); # size of physics
    # slot-4 (read by the backend as app.uinf): Matlab/Julia write `externalparam` here, Python
    # historically wrote `uinf`. M1 reconciliation -> prefer externalparam for cross-frontend
    # consistency, falling back to uinf when externalparam is unset/all-zero (backward-compat).
    # Match MATLAB/Julia: write `externalparam` whenever it is present (an all-zero
    # externalparam is VALID data, not "unset"), falling back to `uinf` only when it is
    # genuinely absent/empty. The previous `(_ep != 0).any()` test made Python serialize a
    # different app.bin than MATLAB/Julia for an intentionally all-zero externalparam.
    _ep = array(app['externalparam']).flatten(order='F') if ('externalparam' in app) else array([]);
    _slot4 = _ep if (_ep.size > 0) else array(app['uinf']).flatten(order='F');
    app['_slot4'] = _slot4;
    nsize[4-1] = size(_slot4); # boundary data (externalparam, uinf fallback)
    nsize[5-1] = size(app['dt']); # number of time steps
    nsize[6-1] = size(app['factor']); # size of factor
    nsize[7-1] = size(app['physicsparam']); # number of physical parameters
    nsize[8-1] = size(app['solversparam']); # number of solver parameters
    nsize[9-1] = size(app['tau']); # number of solver parameters
    nsize[10-1] = size(app['stgdata']);
    nsize[11-1] = size(app['stgparam']);
    nsize[12-1] = size(app['stgib']);
    nsize[13-1] = size(app['vindx']);
    nsize[14-1] = size(app['dae_dt']); # number of dual time steps
    nsize[15-1] = size(flat('interfacefluxmap'));
    nsize[16-1] = size(avparam);
    nsize[17-1] = size(flat('wmModelIDs'));
    nsize[18-1] = size(flat('wmBoundaries'));
    nsize[19-1] = size(flat('wmDistances'));
    nsize[20-1] = size(avfilterparam);
    nsize[21-1] = size(meshadaptparam)
    nsize[22-1] = size(meshadaptbcs)
    nsize[23-1] = size(distanceboundaryconditions)

    print("Writing app into file...");
    fileID = open(filename, 'wb');
    array(size(nsize), dtype=float64).tofile(fileID)
    nsize.astype('float64').tofile(fileID)
    if nsize[1-1] > 0:
        ndims.astype('float64').tofile(fileID);
    if nsize[2-1] > 0:
        app['flag'] = array(app['flag']).flatten(order = 'F');
        app['flag'].astype('float64').tofile(fileID);
    if nsize[3-1] > 0:
        app['problem'] = array(app['problem']).flatten(order = 'F');
        app['problem'].astype('float64').tofile(fileID);
    if nsize[4-1] > 0:
        app['_slot4'].astype('float64').tofile(fileID);  # externalparam (uinf fallback) -- M1
    if nsize[5-1] > 0:
        app['dt'] = array(app['dt']).flatten(order = 'F');
        app['dt'].astype('float64').tofile(fileID);
    if nsize[6-1] > 0:
        app['factor'] = array(app['factor']).flatten(order = 'F');
        app['factor'].astype('float64').tofile(fileID);
    if nsize[7-1] > 0:
        app['physicsparam'] = array(app['physicsparam']).flatten(order = 'F');
        app['physicsparam'].astype('float64').tofile(fileID);
    if nsize[8-1] > 0:
        app['solversparam'] = array(app['solversparam']).flatten(order = 'F');
        app['solversparam'].astype('float64').tofile(fileID);
    if nsize[9-1] > 0:
        app['tau'] = array(app['tau']).flatten(order = 'F');
        app['tau'].astype('float64').tofile(fileID);
    if nsize[10-1] > 0:
        app['stgdata'] = array(app['stgdata']).flatten(order = 'F');
        app['stgdata'].astype('float64').tofile(fileID);
    if nsize[11-1] > 0:
        app['stgparam'] = array(app['stgparam']).flatten(order = 'F');
        app['stgparam'].astype('float64').tofile(fileID);
    if nsize[12-1] > 0:
        app['stgib'] = array(app['stgib']).flatten(order = 'F');
        app['stgib'].astype('float64').tofile(fileID);
    if nsize[13-1] > 0:
        app['vindx'] = array(app['vindx']).flatten(order = 'F')-1;
        app['vindx'].astype('float64').tofile(fileID);
    if nsize[14-1] > 0:
        app['dae_dt'] = array(app['dae_dt']).flatten(order = 'F');
        app['dae_dt'].astype('float64').tofile(fileID);
    if nsize[15-1] > 0:
        app['interfacefluxmap'] = flat('interfacefluxmap') - 1;
        app['interfacefluxmap'].astype('float64').tofile(fileID);
    if nsize[16-1] > 0:
        avparam.astype('float64').tofile(fileID);
    if nsize[17-1] > 0:
        app['wmModelIDs'] = flat('wmModelIDs');
        app['wmModelIDs'].astype('float64').tofile(fileID);
    if nsize[18-1] > 0:
        app['wmBoundaries'] = flat('wmBoundaries');
        app['wmBoundaries'].astype('float64').tofile(fileID);
    if nsize[19-1] > 0:
        app['wmDistances'] = flat('wmDistances');
        app['wmDistances'].astype('float64').tofile(fileID);
    if nsize[20-1] > 0:
        avfilterparam.astype('float64').tofile(fileID);
    if nsize[21-1] > 0:
        meshadaptparam.tofile(fileID)
    if nsize[22-1] > 0:
        meshadaptbcs.tofile(fileID)
    if nsize[23-1] > 0:
        distanceboundaryconditions.tofile(fileID)

    if app['mutationflag']:
        app['mutationopts']['MixtureName'] = array((app['mutationopts']['MixtureName'] +'X').encode())
        app['mutationopts']['MixtureName'].tofile(fileID)
        app['mutationopts']['StateModel'] = array((app['mutationopts']['StateModel'] +'X').encode())
        app['mutationopts']['StateModel'].tofile(fileID)
        app['mutationopts']['ThermoDB'] = array((app['mutationopts']['ThermoDB'] +'X').encode())
        app['mutationopts']['ThermoDB'].tofile(fileID)
        
    fileID.close();

    return app;
