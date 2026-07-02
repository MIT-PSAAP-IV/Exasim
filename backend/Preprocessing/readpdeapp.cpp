/*
    readpdeapp.cpp

    This file provides functionality for parsing PDE application input files, extracting parameters, and initializing PDE data structures for use in numerical simulations.

    Main Components:
    ----------------

    1. InputParams Struct:
        - Holds all parsed input parameters from the PDE application file.
        - Includes maps for std::string, double, and integer parameters.
        - Contains vectors for boundary conditions, physics parameters, and other simulation-specific data.

    2. Helper Functions:
        - parseList<T>: Parses a list of numbers from a std::string buffer enclosed in square brackets.
        - parseStringList: Parses a list of strings from a buffer, extracting quoted strings.
        - trim: Removes leading and trailing whitespace from a std::string.
        - tokenizeBraceList: Tokenizes comma- or whitespace-separated numeric lists, respecting parentheses nesting.
        - parseExpression: Parses a list of doubles, supporting "repeat(value, count)" syntax for repeated values.

    3. parseInputFile:
        - Reads and parses the input file, populating an InputParams struct.
        - Checks for required keys and reports missing parameters.
        - Supports assignment of various parameter types and lists.

    4. printInputParams:
        - Utility to print the contents of an InputParams struct for debugging.

    5. PDE Struct:
        - Represents the main PDE configuration, including file paths, model settings, solver options, and parameter vectors.
        - Used to store all simulation configuration after parsing.

    6. extractCoupledData:
        - Extracts coupled interface, condition, and boundary condition data from input vectors.

    7. initializePDE:
        - Initializes a PDE struct from parsed InputParams.
        - Sets default values, assigns parameters, and configures file paths.
        - Handles logic for hybrid discretization, model selection, and coupled interface extraction.

    8. writepde:
        - Serializes the PDE struct to a binary file for use in downstream simulation components.
        - Writes parameter vectors and configuration data in a specific order.

    Usage:
    ------
    - Use parseInputFile to read and parse a PDE application input file.
    - Use initializePDE to create a PDE struct from parsed parameters.
    - Use writepde to serialize the PDE configuration for simulation.
    - Use printInputParams for debugging and inspection of parsed parameters.

    Notes:
    ------
    - The code expects certain required keys to be present in the input file.
    - Error handling is performed for missing parameters and invalid paths.
    - The code supports flexible input formats, including repeated values and nested lists.

*/
#ifndef __READPDEAPP
#define __READPDEAPP

// Struct to hold all parsed input parameters
// struct InputParams {
//     std::string pdeappfile;
//     std::unordered_map<std::string, std::string> stringParams;
//     std::unordered_map<std::string, double> doubleParams;
//     std::unordered_map<std::string, int> intParams;
// 
//     std::vector<int> interfaceConditions;
//     std::vector<int> interfaceFluxmap;
//     std::vector<int> boundaryConditions;
//     std::vector<int> curvedBoundaries;
//     std::vector<int> periodicBoundaries1;
//     std::vector<int> periodicBoundaries2;
//     std::vector<int> cartGridPart;
// 
//     std::vector<double> dae_dt;
//     std::vector<double> dt;
//     std::vector<double> tau;
//     std::vector<double> physicsParam;
//     std::vector<double> externalParam;
//     std::vector<double> vindx;
//     std::vector<double> avparam1, avparam2;       
//     std::vector<double> stgib;
//     std::vector<double> stgdata;
//     std::vector<double> stgparam;
// 
//     std::vector<std::string> boundaryExprs;    
//     std::vector<std::string> curvedBoundaryExprs;    
//     std::vector<std::string> periodicExprs1;    
//     std::vector<std::string> periodicExprs2;        
// 
//     std::unordered_set<std::string> foundKeys;
// };

// Helper to parse a list of numbers in braces
template <typename T>
std::vector<T> parseList(const std::string& buffer) {
    std::vector<T> result;
    std::regex brace_content("\\[([^\\}]*)\\]");
    std::smatch match;

    if (std::regex_search(buffer, match, brace_content)) {
        std::string content = match[1];
        std::istringstream ss(content);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::istringstream converter(item);
            T value;
            converter >> value;
            result.push_back(value);
        }
    }
    return result;
}

// Helper to parse a list of strings in braces
inline std::vector<std::string> parseStringList(const std::string& buffer) {
    std::vector<std::string> result;
    std::regex string_regex("\"([^\"]*)\"");
    auto begin = std::sregex_iterator(buffer.begin(), buffer.end(), string_regex);
    auto end = std::sregex_iterator();

    for (auto i = begin; i != end; ++i) {
        result.push_back((*i)[1]);
    }
    return result;
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::vector<std::string> tokenizeBraceList(const std::string& input) {
    std::vector<std::string> tokens;
    std::string current;
    int parenLevel = 0;

    for (char ch : input) {
        const bool isSpace = (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r');
        if ((ch == ',' || isSpace) && parenLevel == 0) {
            std::string token = trim(current);
            if (!token.empty())
                tokens.push_back(token);
            current.clear();
        } else {
            if (ch == '(') ++parenLevel;
            else if (ch == ')') --parenLevel;
            current += ch;
        }
    }
    std::string token = trim(current);
    if (!token.empty())
        tokens.push_back(token);
    return tokens;
}

inline std::vector<double> parseExpression(const std::string& expr) {
    std::string content = expr;

    // Extract content between first { and last }
    size_t lbrace = content.find('[');
    size_t rbrace = content.rfind(']');

    if (lbrace == std::string::npos || rbrace == std::string::npos || lbrace >= rbrace) {        
        error("Error: expression must contain square brackets [...]");
    }

    content = content.substr(lbrace + 1, rbrace - lbrace - 1);

    auto tokens = tokenizeBraceList(content);

    std::regex repeatPattern(R"(repeat\s*\(\s*([\d\.eE\+\-]+)\s*,\s*(\d+)\s*\))");

    std::vector<double> result;
    for (const auto& tok : tokens) {
        std::smatch match;
        if (std::regex_match(tok, match, repeatPattern)) {
            double value = std::stod(match[1]);
            int count = std::stoi(match[2]);
            result.insert(result.end(), count, value);
        } else {
            result.push_back(std::stod(tok));
        }
    }

    return result;
}

inline std::vector<double> parseMatrixExpression(const std::string& expr, int& nrows, int& ncols) {
    std::string content = expr;
    size_t lbrace = content.find('[');
    size_t rbrace = content.rfind(']');

    if (lbrace == std::string::npos || rbrace == std::string::npos || lbrace >= rbrace)
        error("Error: physicsparamcases must contain square brackets [...]");

    content = content.substr(lbrace + 1, rbrace - lbrace - 1);
    std::vector<std::vector<double>> rows;

    if (content.find('[') != std::string::npos) {
        size_t pos = 0;
        while (true) {
            size_t rowStart = content.find('[', pos);
            if (rowStart == std::string::npos)
                break;
            size_t rowEnd = content.find(']', rowStart);
            if (rowEnd == std::string::npos)
                error("Malformed physicsparamcases row");
            std::string row = "[" + content.substr(rowStart + 1, rowEnd - rowStart - 1) + "]";
            rows.push_back(parseExpression(row));
            pos = rowEnd + 1;
        }
    }
    else {
        std::stringstream ss(content);
        std::string row;
        while (std::getline(ss, row, ';')) {
            row = trim(row);
            if (!row.empty())
                rows.push_back(parseExpression("[" + row + "]"));
        }
    }

    if (rows.empty())
        error("physicsparamcases must contain at least one row");

    nrows = static_cast<int>(rows.size());
    ncols = static_cast<int>(rows[0].size());
    if (ncols <= 0)
        error("physicsparamcases rows must be nonempty");

    std::vector<double> result;
    result.reserve(static_cast<size_t>(nrows) * ncols);
    for (int i = 0; i < nrows; ++i) {
        if (static_cast<int>(rows[i].size()) != ncols)
            error("All physicsparamcases rows must have the same number of parameters");
        result.insert(result.end(), rows[i].begin(), rows[i].end());
    }
    return result;
}

inline InputParams parseInputFile(const std::string& filename, int mpirank=0)
{
    InputParams params;
    
    params.pdeappfile = filename;
    
    std::ifstream file(filename);
    std::string line, buffer;

    auto markFound = [&](const std::string& key) {
        params.foundKeys.insert(key);
    };

    auto parseAndAssign = [&](const std::string& full) {      
        if (full.find("boundaryconditions") != std::string::npos) {
            params.boundaryConditions = parseList<int>(full);
            markFound("boundaryconditions");
        }
        else if (full.find("boundaryexpressions") != std::string::npos) {
            params.boundaryExprs = parseStringList(full);
            markFound("boundaryexpressions");
        }
        else if (full.find("curvedboundaries") != std::string::npos)
            params.curvedBoundaries = parseList<int>(full);
        else if (full.find("curvedboundaryexprs") != std::string::npos)          
            params.curvedBoundaryExprs = parseStringList(full);
        else if (full.find("periodicboundaries1") != std::string::npos)
            params.periodicBoundaries1 = parseList<int>(full);
        else if (full.find("periodicexprs1") != std::string::npos)
            params.periodicExprs1 = parseStringList(full);
        else if (full.find("periodicboundaries2") != std::string::npos)
            params.periodicBoundaries2 = parseList<int>(full);
        else if (full.find("periodicexprs2") != std::string::npos)
            params.periodicExprs2 = parseStringList(full);
        else if (full.find("physicsparamwarmstart") != std::string::npos) {
            std::regex key_val_num("([a-zA-Z0-9_]+)\\s*=\\s*([^;]+)");
            std::smatch match;
            if (std::regex_search(full, match, key_val_num)) {
                params.intParams[match[1]] = static_cast<int>(std::stod(match[2]));
                markFound(match[1]);
            }
        }
        else if (full.find("physicsparamcases") != std::string::npos) {
            int nrows = 0, ncols = 0;
            params.physicsParamCases = parseMatrixExpression(full, nrows, ncols);
            markFound("physicsparamcases");
        }
        else if (full.find("physicsparam") != std::string::npos) {
            params.physicsParam = parseExpression(full);
            markFound("physicsparam");
        }
        else if (full.find("tau") != std::string::npos) {
            params.tau = parseExpression(full);
            markFound("tau");
        }
        else if (full.find("dt") != std::string::npos) {
            params.dt = parseExpression(full);
        }
        else if (full.find("dae_dt") != std::string::npos) {
            params.dae_dt = parseExpression(full);
        }
        else if (full.find("externalparam") != std::string::npos) 
            params.externalParam = parseExpression(full);
        else if (full.find("vindx") != std::string::npos) 
            params.vindx = parseExpression(full);
        else if (full.find("avparam1") != std::string::npos) 
            params.avparam1 = parseExpression(full);
        else if (full.find("avparam2") != std::string::npos) 
            params.avparam2 = parseExpression(full);
        else if (full.find("stgib") != std::string::npos) 
            params.stgib = parseExpression(full);
        else if (full.find("stgdata") != std::string::npos) 
            params.stgdata = parseExpression(full);
        else if (full.find("stgparam") != std::string::npos) 
            params.stgparam = parseExpression(full);
        else if (full.find("cartgridpart") != std::string::npos)
            params.cartGridPart = parseList<int>(full);
        else if (full.find("interfaceconditions") != std::string::npos)
            params.interfaceConditions = parseList<int>(full);
        else if (full.find("interfacefluxmap") != std::string::npos)
            params.interfaceFluxmap = parseList<int>(full);
        else {
            std::regex key_val_str("([a-zA-Z0-9_]+)\\s*=\\s*\"([^\"]*)\"");
            std::regex key_val_num("([a-zA-Z0-9_]+)\\s*=\\s*([^;]+)");

            std::smatch match;
            if (std::regex_search(full, match, key_val_str)) {
                params.stringParams[match[1]] = match[2];
                markFound(match[1]);
            }
            else if (std::regex_search(full, match, key_val_num)) {
                std::istringstream iss(match[2]);
                double val;
                iss >> val;
                if (match[2].str().find('.') != std::string::npos || match[2].str().find('e') != std::string::npos)
                    params.doubleParams[match[1]] = val;
                else
                    params.intParams[match[1]] = static_cast<int>(val);
                markFound(match[1]);
            }
        }
    };

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        buffer += line;
        if (line.find("}") != std::string::npos || line.find(";") != std::string::npos) {
            parseAndAssign(buffer);
            buffer.clear();
        }
    }

    std::vector<std::string> requiredKeys = {
        "model", "modelfile", "meshfile", "discretization",
        "platform", "mpiprocs", "porder", "pgauss", "physicsparam",
        "tau", "boundaryconditions", "boundaryexpressions"
    };

    bool err = false;
    for (const auto& key : requiredKeys) {
        if (params.foundKeys.find(key) == params.foundKeys.end()) {
            std::cerr << "ERROR: Missing required key: " << key << std::endl;
            err = true;
        }
    }

    if (err) error("Input file is missing required parameters. Exiting.\n");    
    else if (mpirank==0) std::cout << "Finished parseInputFile.\n";
    
    return params;
}

inline void printInputParams(InputParams& params) 
{
    auto printVec = [](const auto& vec, const std::string& name) {
        std::cout << name << ": ";
        for (const auto& v : vec) std::cout << v << " | ";
        std::cout << "\n";
    };

    // Output the contents
    for (const auto& [k, v] : params.stringParams) std::cout << k << ": " << v << "\n";
    for (const auto& [k, v] : params.intParams) std::cout << k << ": " << v << "\n";
    for (const auto& [k, v] : params.doubleParams) std::cout << k << ": " << v << "\n";

    printVec(params.boundaryConditions, "boundaryConditions");
    printVec(params.boundaryExprs, "boundaryExprs");
    printVec(params.curvedBoundaries, "curvedBoundaries");
    printVec(params.curvedBoundaryExprs, "curvedBoundaryExprs");
    printVec(params.periodicBoundaries1, "periodicBoundaries1");
    printVec(params.periodicExprs1, "periodicExprs1");
    printVec(params.periodicBoundaries2, "periodicBoundaries2");
    printVec(params.periodicExprs2, "periodicExprs2");
    printVec(params.dt, "dt");
    printVec(params.dae_dt, "dae_dt");
    printVec(params.tau, "tau");
    printVec(params.physicsParam, "physicsParam");
    printVec(params.physicsParamCases, "physicsParamCases");
    printVec(params.externalParam, "externalParam");
    printVec(params.cartGridPart, "cartGridPart");
    printVec(params.interfaceConditions, "interfaceConditions");
    printVec(params.interfaceFluxmap, "interfaceFluxmap");
}

// struct PDE {
//     std::string discretization = "ldg";
//     std::string exasimpath = "";
//     std::string datapath = "";
//     std::string datainpath = "";
//     std::string dataoutpath = "";
//     std::string platform = "cpu";   
//     std::string model = "ModelD";
//     std::string pdeappfile = "";
//     std::string modelfile = "pdemodel.txt";
//     std::string meshfile = "mesh.bin";
//     std::string xdgfile = "";
//     std::string udgfile = "";
//     std::string vdgfile = "";
//     std::string wdgfile = "";
//     std::string uhatfile = "";
//     std::string partitionfile = "";
// 
//     int gencode = 1; // 1 for code generation, 0 for no code generation
//     int writemeshsol = 1; // 1 for writing mesh solution, 0 for no writing
//     int modelnumber = 0;
//     int mpiprocs = 1;
//     int nd = 1, nc = 1, ncu = 1, ncq = 0, ncp = 0, ncv = 0;
//     int nch = 1, ncx = 1, ncw = 0, nce = 0, np=0, nve=0, ne=0;
//     int nsca=0, nvec=0, nten=0, nsurf=0, nvqoi=0;
//     int neb = 512 * 8;
//     int nfb = 512 * 16;
//     int elemtype = 1;
//     int nodetype = 1;
//     int hybrid = 0;
//     int tdep = 0;
//     int wave = 0;
//     int linearproblem = 0;
//     int subproblem = 0;
//     int debugmode = 0;
//     int stgNmode = 0;
//     int porder = 1;
//     int pgauss = 2;
//     int temporalscheme = 0;
//     int torder = 1;
//     int nstage = 1;
//     int convStabMethod = 0;
//     int diffStabMethod = 0;
//     int rotatingFrame = 0;
//     int viscosityModel = 0;
//     int SGSmodel = 0;
//     int ALE = 0;
//     int AV = 0;
//     int AVdistfunction = 0;
//     int AVsmoothingIter = 2;
//     int frozenAVflag = 1;
//     int nonlinearsolver = 0;
//     int linearsolver = 0;
//     int NewtonIter = 20;
//     int GMRESiter = 200;
//     int GMRESrestart = 25;
//     int GMRESortho = 0;
//     int preconditioner = 0;
//     int precMatrixType = 0;
//     int ppdegree = 0;
//     int NLMatrixType = 0;
//     int runmode = 0;
//     int tdfunc = 1;
//     int sourcefunc = 1;
//     int matvecorder = 1;
//     int RBdim = 5;
//     int saveSolFreq = 1;
//     int saveSolOpt = 1;
//     int timestepOffset = 0;
//     int saveSolBouFreq = 0;
//     int ibs = 0;
//     int compudgavg = 0;
//     int extFhat = 0;
//     int extUhat = 0;
//     int extStab = 0;
//     int saveResNorm = 0;
//     int dae_steps = 0;
// 
//     int coupledinterface = 0; 
//     int coupledcondition = 0;
//     int coupledboundarycondition = 0;    
// 
//     double time = 0.0;
//     double NLparam = 0.0;
//     double NewtonTol = 1e-6;
//     double GMREStol = 1e-3;
//     double matvectol = 1e-3;
//     double dae_alpha = 1.0;
//     double dae_beta = 0.0;
//     double dae_gamma = 0.0;
//     double dae_epsilon = 0.0;    
// 
//     std::vector<int> interfaceFluxmap;    
//     std::vector<double> dae_dt;
//     std::vector<double> dt;
//     std::vector<double> tau;
//     std::vector<double> flag;
//     std::vector<double> problem;
//     std::vector<double> solversparam;
//     std::vector<double> factor;
//     std::vector<double> physicsparam;    
//     std::vector<double> externalparam;            
//     std::vector<double> vindx;
//     std::vector<double> avparam1, avparam2;       
//     std::vector<double> stgib;
//     std::vector<double> stgdata;
//     std::vector<double> stgparam;    
// };

inline void extractCoupledData(
    const std::vector<int>& interfacecondition,
    const std::vector<int>& boundarycondition,
    std::vector<int>& coupledinterface,
    std::vector<int>& coupledcondition,
    std::vector<int>& coupledboundarycondition)
{
    coupledinterface.clear();
    coupledcondition.clear();
    coupledboundarycondition.clear();

    for (size_t i = 0; i < interfacecondition.size(); ++i) {
        if (interfacecondition[i] > 0) {
            coupledinterface.push_back(static_cast<int>(i));
            coupledcondition.push_back(interfacecondition[i]);
            coupledboundarycondition.push_back(boundarycondition[i]);
        }
    }

    if (coupledinterface.empty()) {
        coupledinterface.push_back(0);
        coupledcondition.push_back(0);
        coupledboundarycondition.push_back(0);
    }

    if (coupledinterface.size() > 1) {
        throw std::runtime_error("Error: Multiple coupled interfaces detected.");
    }
}

template<typename... Args>
std::vector<double> makeDoubleVector(Args... args) {
    return { static_cast<double>(args)... };
}

// Pack the user-set scalar fields of `pde` into the runtime-side
// flag/problem/factor/solversparam arrays that downstream code
// (`writepde`, `readsolstruct`, etc.) reads at offset, and apply the
// derived-field rules (hybrid from discretization, tdep from dt).
//
// Called automatically at the tail of `initializePDE` *and* by the
// programmatic CPreprocessing constructor. External apps populating
// the PDE struct directly don't need to call this themselves.
inline void pdeFinalizeDerived(PDE& pde)
{
    if (pde.dt.size() > 0 && pde.dt[0] > 0) pde.tdep = 1;

    if (pde.discretization == "ldg" || pde.discretization == "LDG")
        pde.hybrid = 0;
    else if (pde.discretization == "hdg" || pde.discretization == "HDG")
        pde.hybrid = 1;

    pde.flag = makeDoubleVector(
        pde.tdep, pde.wave, pde.linearproblem, pde.debugmode, pde.matvecorder, pde.GMRESortho,
        pde.preconditioner, pde.precMatrixType, pde.NLMatrixType, pde.runmode, pde.tdfunc, pde.sourcefunc,
        pde.modelnumber, pde.extFhat, pde.extUhat, pde.extStab, pde.subproblem, pde.saveParaview,
        pde.physicsparamwarmstart, pde.builtinmodelID
    );
    pde.problem = makeDoubleVector(
        pde.hybrid, 0, pde.temporalscheme, pde.torder, pde.nstage, pde.convStabMethod,
        pde.diffStabMethod, pde.rotatingFrame, pde.viscosityModel, pde.SGSmodel, pde.ALE, pde.AV,
        pde.linearsolver, pde.NewtonIter, pde.GMRESiter, pde.GMRESrestart, pde.RBdim,
        pde.saveSolFreq, pde.saveSolOpt, pde.timestepOffset, pde.stgNmode, pde.saveSolBouFreq, pde.ibs,
        pde.dae_steps, pde.saveResNorm, pde.AVsmoothingIter, pde.frozenAVflag, pde.ppdegree,
        pde.coupledinterface, pde.coupledcondition, pde.coupledboundarycondition, pde.AVdistfunction
    );
    pde.factor       = {pde.time, pde.dae_alpha, pde.dae_beta, pde.dae_gamma, pde.dae_epsilon};
    pde.solversparam = {pde.NewtonTol, pde.GMREStol, pde.matvectol, pde.NLparam};
}

inline bool useBuiltInAppMetadata(const PDE& pde)
{
    return (pde.builtinmodelID > 0);
}

inline void applyBuiltInABIMetadata(PDE& pde, const ExasimDriverABI& abi)
{
    if (pde.ncu == 0 && abi.ncu > 0)   pde.ncu = abi.ncu;
    if (pde.ncv == 0 && abi.nco > 0)   pde.ncv = abi.nco;
    if (pde.ncw == 0 && abi.ncw > 0)   pde.ncw = abi.ncw;
    if (pde.nsca == 0 && abi.nsca > 0) pde.nsca = abi.nsca;
    if (pde.nvec == 0 && abi.nvec > 0) pde.nvec = abi.nvec;
    if (pde.nten == 0 && abi.nten > 0) pde.nten = abi.nten;
    if (pde.nsurf == 0 && abi.nsurf > 0) pde.nsurf = abi.nsurf;
    if (pde.nvqoi == 0 && abi.nvqoi > 0) pde.nvqoi = abi.nvqoi;
}

inline void applyParsedSpecMetadata(PDE& pde, const ParsedSpec& spec)
{
    for (const auto& vec : spec.vectors) {
        const std::string& name = vec.first;
        int size = vec.second;
        if (name == "uhat") pde.ncu = size;
        if (name == "v")    pde.ncv = size;
        if (name == "w")    pde.ncw = size;
        if (name == "uq")   pde.nc  = size;
    }

    for (size_t i = 0; i < spec.functions.size(); ++i) {
        const auto& fn = spec.functions[i];
        if (fn.name == "VisScalars")  pde.nsca  = fn.outputsize;
        if (fn.name == "VisVectors")  pde.nvec  = fn.outputsize / pde.nd;
        if (fn.name == "VisTensors")  pde.nten  = fn.outputsize / (pde.nd * pde.nd);
        if (fn.name == "QoIboundary") pde.nsurf = fn.outputsize;
        if (fn.name == "QoIvolume")   pde.nvqoi = fn.outputsize;
    }
}

inline void validateBuiltInAppMetadata(const PDE& pde)
{
    if (pde.ncu < 0)
        error("builtinmodelID > 0 requires ncu >= 0 in pdeapp.txt.");
    if (pde.ncv < 0)
        error("builtinmodelID > 0 requires ncv >= 0 in pdeapp.txt.");
    if (pde.ncw < 0)
        error("builtinmodelID > 0 requires ncw >= 0 in pdeapp.txt.");
    if (pde.nsca < 0 || pde.nvec < 0 || pde.nten < 0 || pde.nsurf < 0 || pde.nvqoi < 0)
        error("builtinmodelID > 0 requires nsca, nvec, nten, nsurf, and nvqoi to be nonnegative in pdeapp.txt.");
}

inline void finalizePDEModelSizes(PDE& pde)
{
    if (pde.model=="ModelC" || pde.model=="modelC") {
        pde.wave = 0;
        pde.nc = pde.ncu;
    } else if (pde.model=="ModelD" || pde.model=="modelD") {
        pde.wave = 0;
        pde.nc = pde.ncu * (pde.nd + 1);
    } else if (pde.model=="ModelW" || pde.model=="modelW") {
        pde.tdep = 1;
        pde.wave = 1;
        pde.nc = pde.ncu * (pde.nd + 1);
    }

    if (pde.nc < pde.ncu)
        error("Invalid PDE dimensions: nc must be greater than or equal to ncu.");

    pde.ncq = pde.nc - pde.ncu;
    pde.nch = pde.ncu;
}

inline PDE initializePDE(InputParams& params, int mpirank=0)
{
    PDE pde;
    
    if (params.stringParams.count("exasimpath")) {
        pde.exasimpath = params.stringParams["exasimpath"];
    }
    if (params.stringParams.count("datapath")) {
        pde.datapath = params.stringParams["datapath"];
    }
    if (params.stringParams.count("model")) {
        pde.model = params.stringParams["model"];
    }
    if (params.stringParams.count("modelfile")) {
        pde.modelfile = params.stringParams["modelfile"];
    }
    if (params.stringParams.count("meshfile")) {
        pde.meshfile = params.stringParams["meshfile"];
    }
    if (params.stringParams.count("xdgfile")) {
        pde.xdgfile = params.stringParams["xdgfile"];
    }
    if (params.stringParams.count("udgfile")) {
        pde.udgfile = params.stringParams["udgfile"];
    }
    if (params.stringParams.count("vdgfile")) {
        pde.vdgfile = params.stringParams["vdgfile"];
    }
    if (params.stringParams.count("wdgfile")) {
        pde.wdgfile = params.stringParams["wdgfile"];
    }
    if (params.stringParams.count("uhatfile")) {
        pde.uhatfile = params.stringParams["uhatfile"];
    }
    if (params.stringParams.count("partitionfile")) {
        pde.partitionfile = params.stringParams["partitionfile"];
    }
    if (params.stringParams.count("discretization")) {
        pde.discretization = params.stringParams["discretization"];
    }
    if (params.stringParams.count("platform")) {
        pde.platform = params.stringParams["platform"];
    }

    if (params.intParams.count("builtinmodelID")) {
        pde.builtinmodelID = params.intParams["builtinmodelID"];
    }
    if (params.intParams.count("gendatain")) {
        pde.gendatain = params.intParams["gendatain"];
    }
    if (params.intParams.count("gencode")) {
        pde.gencode = params.intParams["gencode"];
    }
    if (params.intParams.count("writemeshsol")) {
        pde.writemeshsol = params.intParams["writemeshsol"];
    }
    if (params.intParams.count("mpiprocs")) {
        pde.mpiprocs = params.intParams["mpiprocs"];
    }
    if (params.intParams.count("debugmode")) {
        pde.debugmode = params.intParams["debugmode"];
    }
    if (params.intParams.count("runmode")) {
        pde.runmode = params.intParams["runmode"];
    }    
    if (params.intParams.count("modelnumber")) {
        pde.modelnumber = params.intParams["modelnumber"];
    }
    if (params.intParams.count("nodetype")) {
        pde.nodetype = params.intParams["nodetype"];
    }
    if (params.intParams.count("ncu")) {
        pde.ncu = params.intParams["ncu"];
    }
    if (params.intParams.count("ncv")) {
        pde.ncv = params.intParams["ncv"];
    }
    if (params.intParams.count("ncw")) {
        pde.ncw = params.intParams["ncw"];
    }
    if (params.intParams.count("nsca")) {
        pde.nsca = params.intParams["nsca"];
    }
    if (params.intParams.count("nvec")) {
        pde.nvec = params.intParams["nvec"];
    }
    if (params.intParams.count("nten")) {
        pde.nten = params.intParams["nten"];
    }
    if (params.intParams.count("nsurf")) {
        pde.nsurf = params.intParams["nsurf"];
    }
    if (params.intParams.count("nvqoi")) {
        pde.nvqoi = params.intParams["nvqoi"];
    }
    if (params.intParams.count("neb")) {
        pde.neb = params.intParams["neb"];
    }
    if (params.intParams.count("nfb")) {
        pde.nfb = params.intParams["nfb"];
    }
    if (params.intParams.count("linearproblem")) {
        pde.linearproblem = params.intParams["linearproblem"];
    }
    if (params.intParams.count("subproblem")) {
        pde.subproblem = params.intParams["subproblem"];
    }
    if (params.intParams.count("saveParaview")) {
        pde.saveParaview = params.intParams["saveParaview"];
    }
    if (params.intParams.count("physicsparamwarmstart")) {
        pde.physicsparamwarmstart = params.intParams["physicsparamwarmstart"];
    }
    if (params.intParams.count("tdep")) {
        pde.tdep = params.intParams["tdep"];
    }
    if (params.intParams.count("wave")) {
        pde.wave = params.intParams["wave"];
    }
    if (params.intParams.count("torder")) {
        pde.torder = params.intParams["torder"];
    }
    if (params.intParams.count("nstage")) {
        pde.nstage = params.intParams["nstage"];
    }
    if (params.intParams.count("porder")) {
        pde.porder = params.intParams["porder"];
    }
    if (params.intParams.count("pgauss")) {
        pde.pgauss = params.intParams["pgauss"];
    }        
    if (params.intParams.count("stgNmode")) {
        pde.stgNmode = params.intParams["stgNmode"];
    }
    if (params.intParams.count("temporalscheme")) {
        pde.temporalscheme = params.intParams["temporalscheme"];
    }
    if (params.intParams.count("convStabMethod")) {
        pde.convStabMethod = params.intParams["convStabMethod"];
    }
    if (params.intParams.count("diffStabMethod")) {
        pde.diffStabMethod = params.intParams["diffStabMethod"];
    }
    if (params.intParams.count("rotatingFrame")) {
        pde.rotatingFrame = params.intParams["rotatingFrame"];
    }
    if (params.intParams.count("viscosityModel")) {
        pde.viscosityModel = params.intParams["viscosityModel"];
    }
    if (params.intParams.count("SGSmodel")) {
        pde.SGSmodel = params.intParams["SGSmodel"];
    }
    if (params.intParams.count("ALE")) {
        pde.ALE = params.intParams["ALE"];
    }
    if (params.intParams.count("AV")) {
        pde.AV = params.intParams["AV"];
    }
    if (params.intParams.count("AVdistfunction")) {
        pde.AVdistfunction = params.intParams["AVdistfunction"];
    }
    if (params.intParams.count("AVsmoothingIter")) {
        pde.AVsmoothingIter = params.intParams["AVsmoothingIter"];
    }
    if (params.intParams.count("frozenAVflag")) {
        pde.frozenAVflag = params.intParams["frozenAVflag"];
    }
    if (params.intParams.count("nonlinearsolver")) {
        pde.nonlinearsolver = params.intParams["nonlinearsolver"];
    }
    if (params.intParams.count("linearsolver")) {
        pde.linearsolver = params.intParams["linearsolver"];
    }
    if (params.intParams.count("NewtonIter")) {
        pde.NewtonIter = params.intParams["NewtonIter"];
    }
    if (params.intParams.count("GMRESiter")) {
        pde.GMRESiter = params.intParams["GMRESiter"];
    }
    if (params.intParams.count("GMRESrestart")) {
        pde.GMRESrestart = params.intParams["GMRESrestart"];
    }
    if (params.intParams.count("GMRESortho")) {
        pde.GMRESortho = params.intParams["GMRESortho"];
    }
    if (params.intParams.count("preconditioner")) {
        pde.preconditioner = params.intParams["preconditioner"];
    }
    if (params.intParams.count("precMatrixType")) {
        pde.precMatrixType = params.intParams["precMatrixType"];
    }
    if (params.intParams.count("ppdegree")) {
        pde.ppdegree = params.intParams["ppdegree"];
    }
    if (params.intParams.count("NLMatrixType")) {
        pde.NLMatrixType = params.intParams["NLMatrixType"];
    }
    if (params.intParams.count("runmode")) {
        pde.runmode = params.intParams["runmode"];
    }
    if (params.intParams.count("tdfunc")) {
        pde.tdfunc = params.intParams["tdfunc"];
    }
    if (params.intParams.count("sourcefunc")) {
        pde.sourcefunc = params.intParams["sourcefunc"];
    }
    if (params.intParams.count("matvecorder")) {
        pde.matvecorder = params.intParams["matvecorder"];
    }
    if (params.intParams.count("RBdim")) {
        pde.RBdim = params.intParams["RBdim"];
    }
    if (params.intParams.count("saveSolFreq")) {
        pde.saveSolFreq = params.intParams["saveSolFreq"];
    }
    if (params.intParams.count("saveSolOpt")) {
        pde.saveSolOpt = params.intParams["saveSolOpt"];
    }
    if (params.intParams.count("timestepOffset")) {
        pde.timestepOffset = params.intParams["timestepOffset"];
    }
    if (params.intParams.count("saveSolBouFreq")) {
        pde.saveSolBouFreq = params.intParams["saveSolBouFreq"];
    }
    if (params.intParams.count("ibs")) {
        pde.ibs = params.intParams["ibs"];
    }
    if (params.intParams.count("compudgavg")) {
        pde.compudgavg = params.intParams["compudgavg"];
    }
    if (params.intParams.count("extFhat")) {
        pde.extFhat = params.intParams["extFhat"];
    }
    if (params.intParams.count("extUhat")) {
        pde.extUhat = params.intParams["extUhat"];
    }
    if (params.intParams.count("extStab")) {
        pde.extStab = params.intParams["extStab"];
    }
    if (params.intParams.count("saveResNorm")) {
        pde.saveResNorm = params.intParams["saveResNorm"];
    }
    if (params.intParams.count("dae_steps")) {
        pde.dae_steps = params.intParams["dae_steps"];
    }
            
    if (params.doubleParams.count("time")) {
        pde.time = params.doubleParams["time"];
    }
    if (params.doubleParams.count("NLparam")) {
        pde.NLparam = params.doubleParams["NLparam"];
    }
    if (params.doubleParams.count("NewtonTol")) {
        pde.NewtonTol = params.doubleParams["NewtonTol"];
    }
    if (params.doubleParams.count("GMREStol")) {
        pde.GMREStol = params.doubleParams["GMREStol"];
    }
    if (params.doubleParams.count("matvectol")) {
        pde.matvectol = params.doubleParams["matvectol"];
    }
    if (params.doubleParams.count("dae_alpha")) {
        pde.dae_alpha = params.doubleParams["dae_alpha"];
    }
    if (params.doubleParams.count("dae_beta")) {
        pde.dae_beta = params.doubleParams["dae_beta"];
    }
    if (params.doubleParams.count("dae_gamma")) {
        pde.dae_gamma = params.doubleParams["dae_gamma"];
    }
    if (params.doubleParams.count("dae_epsilon")) {
        pde.dae_epsilon = params.doubleParams["dae_epsilon"];
    }
     
    pde.dt = params.dt;
    pde.dae_dt = params.dae_dt;
    pde.tau = params.tau;
    pde.physicsparam = params.physicsParam;
    pde.physicsparamcases = params.physicsParamCases;
    if (!pde.physicsparamcases.empty()) {
        if (pde.physicsparam.empty())
            error("physicsparamcases requires a nonempty physicsparam vector");
        if ((pde.physicsparamcases.size() % pde.physicsparam.size()) != 0)
            error("Each physicsparamcases row must have the same length as physicsparam");
    }
    pde.externalparam = params.externalParam;
    pde.vindx = params.vindx;
    pde.avparam1 = params.avparam1;
    pde.avparam2 = params.avparam2;
    pde.stgib = params.stgib;
    pde.stgdata = params.stgdata;
    pde.stgparam = params.stgparam;
    
    if (pde.dt.size() > 0) {if (pde.dt[0]> 0) pde.tdep = 1;}    
    
  
    if (pde.discretization == "ldg" || pde.discretization == "LDG")
      pde.hybrid = 0;
    else if (pde.discretization == "hdg" || pde.discretization == "HDG")
      pde.hybrid = 1;
            
//     if (pde.model=="ModelC" || pde.model=="modelC") {
//         pde.wave = 0;
//         pde.nc = pde.ncu;
//     } else if (pde.model=="ModelD" || pde.model=="modelD") {     
//         pde.wave = 0;
//         pde.nc = (pde.ncu)*(pde.nd+1);
//     } else if (pde.model=="ModelW" || pde.model=="modelW") {
//         pde.tdep = 1;
//         pde.wave = 1;
//         pde.nc = (pde.ncu)*(pde.nd+1);
//     }
//     pde.ncq = pde.nc - pde.ncu;
//     pde.nch  = pde.ncu;               
        
    pde.interfaceFluxmap = params.interfaceFluxmap;    
    if (params.interfaceConditions.size()>0) {
        std::vector<int> a, b, c;
        extractCoupledData(params.interfaceConditions, params.boundaryConditions, a, b, c);
        pde.coupledinterface = a[0]+1;
        pde.coupledcondition = b[0];
        pde.coupledboundarycondition = c[0];            
    }
            
    pdeFinalizeDerived(pde);
    
                    
    // Normalize a path that contains an "Exasim" component down to the Exasim
    // root, but honor an explicitly-set exasimpath verbatim even when its
    // directory name does not literally contain "Exasim" (the checkout may live
    // at e.g. /data/scratch/.../exasim-teoc). Only fall back to the cwd when no
    // exasimpath was provided at all. (Mirrors text2code/readpdeapp.cpp.)
    {
      const std::string rawpath = pde.exasimpath;
      const std::string trimmed = trimToSubstringAtLastOccurence(rawpath, "Exasim");
      if (!trimmed.empty()) {
        pde.exasimpath = trimmed;
      } else if (!rawpath.empty()) {
        pde.exasimpath = rawpath;   // honor an explicit path verbatim
      } else {
        if (mpirank==0) std::cout<<"exasimpath is not set in "<< params.pdeappfile <<" file.\nWe use the working directory to define exasimpath.\n";
        std::filesystem::path cwd = std::filesystem::current_path();
        std::string cwdtrim = trimToSubstringAtLastOccurence(cwd, "Exasim");
        pde.exasimpath = cwdtrim.empty() ? cwd.string() : cwdtrim;
        if (pde.exasimpath == "")
          error("exasimpath is not valid. Please set exasimpath to the correct path of the Exasim source code in pdeapp.txt file.");
      }
    }
    if (mpirank==0) std::cout << "exasimpath = "<<pde.exasimpath<<std::endl;
    
    pde.pdeappfile = params.pdeappfile;    
    if (pde.datapath == "") {      
      std::string dp = trim_dir(params.pdeappfile);
      if (dp == "") {
        if (mpirank==0) std::cout<<"datapath is not set in "<< params.pdeappfile <<".\nWe set datapath to the working directory.\n";
        pde.datapath = std::filesystem::current_path();    
      }
      else {
        if (mpirank==0) std::cout<<"datapath is not set in the input file "<< params.pdeappfile <<".\nWe set datapath by using the path of the input file.\n";
        pde.datapath = dp;    
      }
    }        
    if (mpirank==0) std::cout << "datapath = "<<pde.datapath<<std::endl;
        
    if (pde.modelnumber<=0) {
      //pde.datainpath = make_path(pde.exasimpath, "build/datain");
      //pde.dataoutpath = make_path(pde.exasimpath, "build/dataout");
      pde.datainpath = make_path(pde.datapath, "datain");
      pde.dataoutpath = make_path(pde.datapath, "dataout");
    } else if (pde.modelnumber>0) {
      //pde.datainpath = make_path(pde.exasimpath, "build/datain" + std::to_string(pde.modelnumber));
      //pde.dataoutpath = make_path(pde.exasimpath, "build/dataout" + std::to_string(pde.modelnumber));
      pde.datainpath = make_path(pde.datapath, "datain" + std::to_string(pde.modelnumber));
      pde.dataoutpath = make_path(pde.datapath, "dataout" + std::to_string(pde.modelnumber));
    }
    // HOT.7.14 — let pdeapp override datainpath / dataoutpath. Used
    // by validate_codegen.sh to give each parallel ctest gate its
    // own dataout dir so codegen:X and facade:X don't collide.
    // Relative paths resolve against datapath; absolute paths win.
    if (params.stringParams.count("datainpath")) {
        std::string p = params.stringParams["datainpath"];
        pde.datainpath = (!p.empty() && p[0] == '/') ? p : make_path(pde.datapath, p);
    }
    if (params.stringParams.count("dataoutpath")) {
        std::string p = params.stringParams["dataoutpath"];
        pde.dataoutpath = (!p.empty() && p[0] == '/') ? p : make_path(pde.datapath, p);
    }
    
    if (mpirank==0) std::cout << "Finished initializePDE.\n";
    
    if (pde.debugmode==1) printInputParams(params);
    
    return pde;
}

inline void writepde(const PDE& pde, const std::string& filename) 
{    
    std::vector<double> avparam;
    avparam.insert(avparam.end(), pde.avparam1.begin(), pde.avparam1.end());
    avparam.insert(avparam.end(), pde.avparam2.begin(), pde.avparam2.end());

    std::vector<double> ndims(40, 0.0);
    ndims[0] = pde.mpiprocs;
    ndims[1] = pde.nd;
    ndims[5] = pde.nc;
    ndims[6] = pde.ncu;
    ndims[7] = pde.ncq;
    ndims[8] = pde.ncp;
    ndims[9] = pde.ncv;
    ndims[10] = pde.nch;
    ndims[11] = pde.ncx;
    ndims[12] = pde.nce;
    ndims[13] = pde.ncw;
    ndims[14] = pde.nsca;
    ndims[15] = pde.nvec;
    ndims[16] = pde.nten;
    ndims[17] = pde.nsurf;
    ndims[18] = pde.nvqoi;

    std::vector<double> nsize(30, 0.0);
    nsize[0] = ndims.size();
    nsize[1] = pde.flag.size();
    nsize[2] = pde.problem.size();
    nsize[3] = pde.externalparam.size();
    nsize[4] = pde.dt.size();
    nsize[5] = pde.factor.size();
    nsize[6] = pde.physicsparam.size();
    nsize[7] = pde.solversparam.size();
    nsize[8] = pde.tau.size();
    nsize[9] = pde.stgdata.size();
    nsize[10] = pde.stgparam.size();
    nsize[11] = pde.stgib.size();
    nsize[12] = pde.vindx.size();
    nsize[13] = pde.dae_dt.size();
    nsize[14] = pde.interfaceFluxmap.size();
    nsize[15] = avparam.size();

    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing.");

    auto writeVector = [&](const std::vector<double>& vec) {
        file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    };

    double len = (double)nsize.size();
    file.write(reinterpret_cast<const char*>(&len), sizeof(double));
    if (!nsize.empty()) writeVector(nsize);
    if (!ndims.empty()) writeVector(ndims);
    if (!pde.flag.empty()) writeVector(pde.flag);
    if (!pde.problem.empty()) writeVector(pde.problem);
    if (!pde.externalparam.empty()) writeVector(pde.externalparam);
    if (!pde.dt.empty()) writeVector(pde.dt);
    if (!pde.factor.empty()) writeVector(pde.factor);
    if (!pde.physicsparam.empty()) writeVector(pde.physicsparam);
    if (!pde.solversparam.empty()) writeVector(pde.solversparam);
    if (!pde.tau.empty()) writeVector(pde.tau);
    if (!pde.stgdata.empty()) writeVector(pde.stgdata);
    if (!pde.stgparam.empty()) writeVector(pde.stgparam);
    if (!pde.stgib.empty()) writeVector(pde.stgib);

    if (!pde.vindx.empty()) {
        std::vector<double> vindx(pde.vindx.begin(), pde.vindx.end());
        for (auto& v : vindx) v -= 1;
        writeVector(vindx);
    }

    if (!pde.dae_dt.empty()) writeVector(pde.dae_dt);

    if (!pde.interfaceFluxmap.empty()) {
        std::vector<double> map(pde.interfaceFluxmap.begin(), pde.interfaceFluxmap.end());
        for (auto& v : map) v -= 1;
        writeVector(map);
    }

    if (!avparam.empty()) {
        writeVector(avparam);
    }

    file.close();
    if (!pde.physicsparamcases.empty()) {
        const size_t nparam = pde.physicsparam.size();
        const size_t ncases = pde.physicsparamcases.size() / nparam;
        std::filesystem::path sweepfile = std::filesystem::path(filename).parent_path() / "physicsparamcases.bin";
        std::ofstream sweep(sweepfile.string(), std::ios::binary);
        if (!sweep) throw std::runtime_error("Cannot open physicsparamcases.bin for writing.");
        double header[2] = {static_cast<double>(ncases), static_cast<double>(nparam)};
        sweep.write(reinterpret_cast<const char*>(header), sizeof(header));
        sweep.write(reinterpret_cast<const char*>(pde.physicsparamcases.data()),
                    pde.physicsparamcases.size() * sizeof(double));
        sweep.close();
        std::cout << "Finished writing physicsparam sweep to " << sweepfile.string() << std::endl;
    }
    std::cout << "Finished writing pde to " << filename << std::endl;
}
    
#endif
