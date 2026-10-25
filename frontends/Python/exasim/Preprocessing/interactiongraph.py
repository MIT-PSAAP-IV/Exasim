import numpy as np
from importlib import import_module
from .facenumbering import facenumbering

_FIELDS = {"domains", "type", "interface1", "condition1", "fluxmap1",
           "interface2", "condition2", "fluxmap2"}

def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

def interactiongraph(pde, mesh, interaction):
    n = len(pde) if isinstance(pde, (list, tuple)) else 0
    if n < 2 or not isinstance(mesh, (list, tuple)) or len(mesh) != n:
        raise ValueError("interaction graph requires matching pde/mesh lists with at least two domains")
    if not isinstance(interaction, (list, tuple)) or not interaction:
        raise ValueError("interaction must be a nonempty list of dictionaries")
    if any(np.any(np.asarray(item.get("interfacecondition", [])).reshape(-1) > 0) for item in mesh):
        raise ValueError("generalized interactions cannot be combined with mesh.interfacecondition")
    selectors = [[] for _ in range(n)]; widths = [0] * n
    nfint = [1] * n; maxcondition = [0] * n; records = []
    face_labels = []
    for domain in range(n):
        nd = int(np.asarray(mesh[domain]['p']).shape[0])
        nve = int(np.asarray(mesh[domain]['t']).shape[0])
        elemtype = int((nd == 2 and nve == 4) or (nd == 3 and nve == 8))
        face_labels.append(facenumbering(mesh[domain]['p'], mesh[domain]['t'],
            elemtype, mesh[domain]['boundaryexpr'],
            mesh[domain].get('periodicexpr', []))[0])
    for ordinal, entry in enumerate(interaction, start=1):
        if not isinstance(entry, dict) or set(entry) != _FIELDS:
            raise ValueError(f"interaction entry {ordinal} must contain exactly the eight documented fields")
        domains = entry["domains"]
        if (not isinstance(domains, (list, tuple, np.ndarray)) or len(domains) != 2 or
            not all(_integer(v) for v in domains) or min(domains) < 1 or
            max(domains) > n or domains[0] == domains[1]):
            raise ValueError(f"interaction entry {ordinal} field domains is invalid")
        if not _integer(entry["type"]) or entry["type"] != 1:
            raise ValueError(f"interaction entry {ordinal}: generalized interactions support only type=1")
        endpoints = []; maps = []
        for side in (1, 2):
            domain = int(domains[side-1]) - 1
            selector = entry[f"interface{side}"]; condition = entry[f"condition{side}"]
            fluxmap = np.asarray(entry[f"fluxmap{side}"]).reshape(-1).tolist()
            boundaryexpr = mesh[domain].get("boundaryexpr", [])
            if not _integer(selector) or selector < 1 or selector > len(boundaryexpr):
                raise ValueError(f"interaction entry {ordinal} field interface{side} is invalid")
            if selector in np.asarray(mesh[domain].get("periodicboundary", [])).reshape(-1).tolist():
                raise ValueError(f"interaction entry {ordinal} field interface{side} selects a periodic boundary")
            if not np.any(face_labels[domain] == selector):
                raise ValueError(f"interaction entry {ordinal} field interface{side} selects no boundary face")
            boundary = np.asarray(mesh[domain].get("boundarycondition", [])).reshape(-1).tolist()
            if selector > len(boundary) or boundary[selector-1] <= 0 or boundary.count(boundary[selector-1]) != 1:
                raise ValueError(f"interaction entry {ordinal} field interface{side} has a nonunique boundary condition")
            if selector in selectors[domain]:
                raise ValueError(f"interaction entry {ordinal} field interface{side} overlaps another interaction")
            if not _integer(condition) or condition < 1:
                raise ValueError(f"interaction entry {ordinal} field condition{side} must be a positive integer")
            ncu = int(pde[domain].get("ncu", 0))
            if not fluxmap or not all(_integer(v) and 1 <= v <= ncu for v in fluxmap):
                raise ValueError(f"interaction entry {ordinal} field fluxmap{side} is invalid")
            if not hasattr(import_module(pde[domain]['modelfile']), 'fint'):
                raise ValueError(f"interaction entry {ordinal} domain {domain+1} does not define fint")
            selectors[domain].append(int(selector)); maxcondition[domain] = max(maxcondition[domain], int(condition))
            maps.append(fluxmap); endpoints.append([selector, condition, len(fluxmap)] + [v-1 for v in fluxmap])
        for side in (0, 1):
            domain = int(domains[side]) - 1; width = len(maps[1-side])
            if widths[domain] not in (0, width):
                raise ValueError(f"interaction entry {ordinal} violates the one fint output width per domain rule")
            widths[domain] = width; nfint[domain] = width
        length = 9 + len(maps[0]) + len(maps[1])
        records.extend([length, domains[0]-1, domains[1]-1, 1] + endpoints[0] + endpoints[1])
    controls = ("temporalscheme", "torder", "nstage", "dt", "NLiter", "NLtol",
                "linearsolver", "linearsolvertol", "GMRESrestart", "GMRESortho",
                "linearsolveriter", "preconditioner")
    for key in controls:
        if key in pde[0] and any(key in item and not np.array_equal(item[key], pde[0][key]) for item in pde[1:]):
            raise ValueError(f"interaction graph requires identical solver control {key} on every domain")
    table = np.asarray([1, n, len(interaction)] + records, dtype=np.float64)
    if np.any(table >= 2**31):
        raise ValueError("interaction table value exceeds the supported 32-bit range")
    return table, selectors, nfint, maxcondition
