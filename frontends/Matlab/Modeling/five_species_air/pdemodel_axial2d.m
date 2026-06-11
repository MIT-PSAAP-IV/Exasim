function pde = pdemodel
    pde.mass = @mass;
    pde.flux = @flux;
    pde.source = @source;
    pde.fbouhdg = @fbouhdg;
    pde.fbou = @fbou;
    pde.ubou = @ubou;
    pde.initu = @initu;
    % pde.avfield = @avfield;
    pde.sourcew = @sourcew;
    pde.initw = @initw;
    pde.eos = @eos;
end

function w0 = initw(x, mu, eta)
    % should not be used, overwrite by mesh.wdg 
    w0 = sym(zeros(1,1));
    w0(1) = 1;
end

function m = mass(u, q, w, v, x, t, mu, eta)
    ns = 5;
    ndim = 2;
    m = sym(ones(ns + ndim + 1, 1));
end

function f = flux(u, q, w, v, x, t, mu, eta)
    f = fluxaxial2d(u, q, w, v, x, t, mu, eta);
end

function s = source(u, q, w, v, x, t, mu, eta)
    s = sourceaxial2d(u, q, w, v, x, t, mu, eta);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    ub = sym(ones(8, 1));
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    fb = sym(ones(8, 1));
end

function u0 = initu(x, mu, eta)
    u0 = sym(ones(8, 1));
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    ns = 5;
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);
    cp_scale    = eta(7);
    L_scale     = eta(8);

    uinf = initu(x, mu, eta);

    % uinf = [0.000000650000000   0.228300980000000   0.010260100000000   0.754307040000000   0.007131230000000   0.996653279785810 0   0.607507866832124];
    uinf = uinf(:);

    f_out = (u - uhat);
    f_in = (uinf - uhat);

    % wall boundary condition    
    un = u(ns+1).*n(1) + u(ns+2).*n(2);  
    ui = u;
    ui(ns+1) = ui(ns+1) - n(1).*un;
    ui(ns+2) = ui(ns+2) - n(2).*un;
    fh = ui - uhat;


    %%% Isothermal wall
    [species_thermo_structs, Mw, RU] = thermodynamicsModels();
    
    rho_i_wall = u(1:ns);
    rho_i_wall_dim = rho_i_wall * rho_scale;
    T_wall = eta(12);

    p_wall = pressure(T_wall, rho_i_wall_dim, Mw);
    X_wall = X_i(rho_i_wall_dim, Mw);
    e_dim = mixtureEnergyMass(T_wall, p_wall, X_wall, Mw, species_thermo_structs);
    rhoE_dim = (sum(rho_i_wall_dim) * e_dim);
    rhoE_wall = rhoE_dim / rhoe_scale;

    uf = u;
    uf(6:7) = 0;
    f_iso = uf - uhat;
    f_iso(8) = rhoE_wall - uhat(8);

    %%% Noncatalytic wall
    % set dY/dn = 0
    % Step 1: get pressure from flow (dP/dn = 0)
    T_flow     = w(1) * T_scale;
    rho_i_flow = u(1:5)*rho_scale;
    p_flow = pressure(T_flow, rho_i_flow, Mw);

    % Step 2: get mass fractions from flow (dY/dn = 0)
    Y_i_flow = Y_i(rho_i_flow);

    % [rho_i_noncat, e_noncat] = density_energy_from_YiTP(Y_i_flow, T_wall, p_flow);
    % rhoE_noncat = sum(rho_i_noncat) * e_noncat;
    X_i_flow = (Y_i_flow .* Mw) / sum(Y_i_flow .* Mw);

    % Step 3: set state with Y, P, T_wall (double check how this is done) - I think really just need density
    rho_noncat = density(T_wall, p_flow, X_i_flow, Mw);
    rho_i_noncat = rho_noncat * Y_i_flow;
    e_dim_noncat = mixtureEnergyMass(T_wall, p_flow, X_i_flow, Mw, species_thermo_structs);
    rhoE_dim_noncat = (sum(rho_i_noncat) * e_dim_noncat);
    rhoE_noncat = rhoE_dim_noncat;
    u_noncat = 0*uhat;
    u_noncat(1:5) = rho_i_noncat / rho_scale;
    u_noncat(6:7) = 0;
    u_noncat(8) = rhoE_noncat / rhoe_scale;
    f_noncat = u_noncat - uhat;

    %%% Noncatalytic wall use uh
    rho_i_flow_uh = uhat(1:5)*rho_scale;
    p_flow_uh = pressure(T_flow, rho_i_flow_uh, Mw);

    % Step 2: get mass fractions from flow (dY/dn = 0)
    Y_i_flow_uh = Y_i(rho_i_flow_uh);
    [rho_i_noncat_uh, e_noncat_uh] = density_energy_from_YiTP(Y_i_flow_uh, T_wall, p_flow_uh);
    rhoE_noncat_uh = sum(rho_i_noncat_uh) * e_noncat_uh;

    u_noncat_uh = 0*uhat;
    u_noncat_uh(1:5) = rho_i_noncat_uh / rho_scale;
    u_noncat_uh(6:7) = 0;
    u_noncat_uh(8) = rhoE_noncat_uh / rhoe_scale;
    f_noncat_uh = u_noncat_uh - uhat;

    %%% Supercatalytic wall: u
    % specify Y_i to Y_eq
    % Step 1: get pressure from flow (dP/dn = 0): computed above
    % Step 2: specify mass fractions
    Y_i_cat = [0; 0; 0; 0.7624; 1.0-0.7624];
    X_i_cat = (Y_i_cat .* Mw) / sum(Y_i_cat .* Mw);
    rho_flow = sum(rho_i_flow);
    rho_i_supercat = rho_flow * Y_i_cat;
    e_dim_supercat = mixtureEnergyMass(T_wall, p_flow, X_i_cat, Mw, species_thermo_structs);
    % rhoE_dim_cat = (sum(rho_i_cat) * e_dim_cat);
    % Step 4: set state with Y, P, T_wall (double check how this is done)
    % rho_cat = density(T_wall, p_flow, X_i_cat, Mw);
    % rho_i_cat = rho_cat * Y_i_cat;
    % e_dim_cat = mixtureEnergyMass(T_wall, p_flow, X_i_cat, Mw, species_thermo_structs);
    % rhoE_dim_cat = (sum(rho_i_cat) * e_dim_cat);
    % [rho_i_supercat, e_supercat] = density_energy_from_YiTP(Y_i_cat, T_wall, p_flow);
    rhoE_cat = sum(rho_i_supercat)*e_dim_supercat;
    u_cat = 0*uhat;
    u_cat(1:5) = rho_i_supercat / rho_scale;
    u_cat(6:7) = 0;
    u_cat(8) = rhoE_cat / rhoe_scale;
    f_cat = u_cat - uhat;

    %%% Supercatalytic wall: uh
    % specify Y_i to Y_eq
    % Step 1: get pressure from flow (dP/dn = 0): computed above
    % Step 2: specify mass fractions
    Y_i_cat = [0; 0; 0; 0.7624; 1.0-0.7624];
    % X_i_cat = (Y_i_cat .* Mw) / sum(Y_i_cat .* Mw);

    % Step 4: set state with Y, P, T_wall (double check how this is done)
    % rho_cat = density(T_wall, p_flow, X_i_cat, Mw);
    % rho_i_cat = rho_cat * Y_i_cat;
    % e_dim_cat = mixtureEnergyMass(T_wall, p_flow, X_i_cat, Mw, species_thermo_structs);
    % rhoE_dim_cat = (sum(rho_i_cat) * e_dim_cat);
    [rho_i_supercat_uh, e_supercat_uh] = density_energy_from_YiTP(Y_i_cat, T_wall, p_flow_uh);
    rhoE_cat_uh = sum(rho_i_supercat_uh)*e_supercat_uh;
    u_cat = 0*uhat;
    u_cat(1:5) = rho_i_supercat_uh / rho_scale;
    u_cat(6:7) = 0;
    u_cat(8) = rhoE_cat_uh / rhoe_scale;
    f_cat_uh = u_cat - uhat;



        %%% Partially catalytic wall
    % J_i n = w_cat
    % Step 1: evaluate catalytic source term
    % Step 2: enforce viscous flux of continuity to 0
    % Step 3: grab wall temperature
    gam_i = eta(13:17);
    w_1 = T_wall / T_scale;
    uf = u;
    uf(8) = rhoE_wall;
    uf(6:7) = 0;

    f_species   = flux_visc_species(u, q, w, v, x, t, mu, eta);  
    f_species_iso   = flux_visc_species(uf, q, w_1, v, x, t, mu, eta);  

    fn_species = f_species(:,1)*n(1)+f_species(:,2)*n(2);
    fn_species_iso = f_species_iso(:,1)*n(1)+f_species_iso(:,2)*n(2);
    J_cat = 0*fn_species;
    for is = 1:5
        J_cat(is) = gam_i(is) * sqrt(T_wall / (2*pi)) * sqrt(RU ./ Mw(is)) .* u(is) * rho_scale; %TODO: some ambiguity here...
    end
    fn_species_iso(1:5) = fn_species_iso(1:5) + J_cat / (rho_scale*u_scale);

    f_cat_gam      = f_iso;
    % f_cat_gam(1:5) = f_cat_gam(1:5) + fn_species(1:5); %+ tau * (u(1:5) - uhat(1:5));
    f_cat_gam(1:5) = fn_species - fn_species_iso + tau *  (u(1:5) - uhat(1:5));

    % f_cat_gam      = f_noncat;
    % f_cat_gam(1:2) = f_cat_gam(1:2) - J_cat(1:2) / (rho_scale*u_scale);

    f_noncat_noflux       = f_noncat;
    f_noncat_noflux(1:5)  = f_species(:,1)*n(1)+f_species(:,2)*n(2) + tau * (u(1:5)-uhat(1:5));

    q = q(:);
    f_grad = q(1:8)*n(1) + q(9:16)*n(2) + tau*(u(:) - uhat(:));


    % supsersonic inflow, supersonic outflow, isothermal, noncat, supercat, partial cat  %inv. wall  no flux
    % fb = [f_in               f_out              f_iso    f_noncat f_cat      f_cat_gam     fh         f_grad];
    fb = [f_in               f_out              f_iso    f_noncat f_cat      f_cat_gam       fh         f_grad      f_noncat_uh   f_noncat_noflux    f_cat_uh];
end

function f = eos(u, q, w, v, x, t, mu, eta)
    f = eosnd(u, q, w, v, x, t, mu, eta);    
end

function f = sourcew(u, q, w, v, x, t, mu, eta)
    f = eosnd(u, q, w, v, x, t, mu, eta);        
end
