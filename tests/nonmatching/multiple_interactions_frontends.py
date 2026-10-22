#!/usr/bin/env python3
"""Phase 1 Python validator and golden-table driver (no code generation or solve)."""
import copy
import importlib.util
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root / "frontends" / "Python"))
from exasim.Preprocessing.interactiongraph import interactiongraph
from exasim.Preprocessing.elementpartitionhdg import elementpartitionhdg

model_dir = output.parent
(model_dir / "phase1_model.py").write_text("def fint(*args):\n    return [0.0]\n")
(model_dir / "phase1_no_fint.py").write_text("def flux(*args):\n    return [0.0]\n")
sys.path.insert(0, str(model_dir))

controls = dict(temporalscheme=0,torder=1,nstage=1,dt=np.array([0.0]),NLiter=20,
    NLtol=1e-8,linearsolver=0,linearsolvertol=1e-8,GMRESrestart=25,
    GMRESortho=0,linearsolveriter=200,preconditioner=0)
pde = [dict(controls,ncu=1,elemtype=1,modelfile="phase1_model") for _ in range(3)]
p = np.array([[0.,1.,1.,0.],[0.,0.,1.,1.]])
t = np.array([[0],[1],[2],[3]],dtype=np.int64)
expr = [lambda x: np.abs(x[1])<1e-12, lambda x: np.abs(x[0]-1)<1e-12,
        lambda x: np.abs(x[1]-1)<1e-12, lambda x: np.abs(x[0])<1e-12]
mesh = [dict(p=p,t=t,boundaryexpr=expr,periodicexpr=[],periodicboundary=np.array([],dtype=int),
             boundarycondition=np.array([1,2,3,4]),interfacecondition=np.array([])) for _ in range(3)]
edge1 = dict(domains=[1,2],type=1,interface1=2,condition1=1,fluxmap1=[1],
             interface2=4,condition2=1,fluxmap2=[1])
edge2 = dict(domains=[2,3],type=1,interface1=2,condition1=1,fluxmap1=[1],
             interface2=4,condition2=1,fluxmap2=[1])
interactions = [edge1,edge2]
golden = np.array([1,3,2,11,0,1,1,2,1,1,0,4,1,1,0,
                   11,1,2,1,2,1,1,0,4,1,1,0],dtype=np.float64)
table,_,_,_ = interactiongraph(pde,mesh,interactions)
assert table.dtype == np.float64 and np.array_equal(table,golden)
table.tofile(output)

tri_mesh=copy.deepcopy(mesh)
tri_p=np.column_stack((p,np.array([0.5,0.5])))
tri_t=np.array([[0,1,2,3],[1,2,3,0],[4,4,4,4]],dtype=np.int64)
for item in tri_mesh: item["p"]=tri_p; item["t"]=tri_t
assert np.array_equal(interactiongraph(pde,tri_mesh,interactions)[0],golden)
tet_mesh=copy.deepcopy(mesh)
tet_p=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
tet_t=np.array([[0],[1],[2],[3]],dtype=np.int64)
tet_expr=[lambda x: np.abs(x[0])<1e-12,lambda x: np.abs(x[1])<1e-12,
          lambda x: np.abs(x[2])<1e-12,lambda x: np.abs(np.sum(x,axis=0)-1)<1e-12]
for item in tet_mesh:
    item["p"]=tet_p; item["t"]=tet_t; item["boundaryexpr"]=tet_expr
assert np.array_equal(interactiongraph(pde,tet_mesh,interactions)[0],golden)

def reject(label,expected,pdes=None,meshes=None,edges=None):
    try:
        interactiongraph(pdes or pde,meshes or mesh,edges or interactions)
    except Exception as exc:
        assert expected in str(exc), f"{label}: wrong diagnostic: {exc}"
        return
    raise AssertionError(f"validator accepted rule mutation: {label}")

bad=copy.deepcopy(interactions); del bad[0]["condition1"]; reject("rule1","exactly the eight",edges=bad)
bad=copy.deepcopy(interactions); bad[0]["domains"]=[1,1]; reject("rule2","domains is invalid",edges=bad)
bad=copy.deepcopy(interactions); bad[0]["type"]=0; reject("rule3","only type=1",edges=bad)
badmesh=copy.deepcopy(mesh); badmesh[0]["boundaryexpr"]=expr+[lambda x: np.zeros(x.shape[1],dtype=bool)]; badmesh[0]["boundarycondition"]=np.arange(1,6); bad=copy.deepcopy(interactions); bad[0]["interface1"]=5; reject("rule4","selects no boundary face",meshes=badmesh,edges=bad)
badmesh=copy.deepcopy(mesh); badmesh[0]["boundarycondition"]=np.array([1,2,3,2]); reject("rule5","nonunique boundary condition",meshes=badmesh)
bad=copy.deepcopy(interactions); bad[1]["interface1"]=4; reject("rule6","overlaps another interaction",edges=bad)
bad=copy.deepcopy(interactions); bad[0]["condition1"]=0; reject("rule7","positive integer",edges=bad)
bad=copy.deepcopy(interactions); bad[0]["fluxmap1"]=[]; reject("rule8-empty","fluxmap1 is invalid",edges=bad)
bad=copy.deepcopy(interactions); bad[0]["fluxmap1"]=[2]; reject("rule8-range","fluxmap1 is invalid",edges=bad)
badpde=copy.deepcopy(pde); badpde[0]["modelfile"]="phase1_no_fint"; reject("rule9","does not define fint",pdes=badpde)
badmesh=copy.deepcopy(mesh); badmesh[0]["interfacecondition"]=np.array([0,1,0,0]); reject("rule10","cannot be combined",meshes=badmesh)
badpde=copy.deepcopy(pde); badpde[1]["torder"]=2; reject("rule11-torder","solver control torder",pdes=badpde)
badpde=copy.deepcopy(pde); badpde[1]["nstage"]=2; reject("rule11-nstage","solver control nstage",pdes=badpde)
badmesh=copy.deepcopy(mesh); badmesh[0]["periodicboundary"]=np.array([2]); reject("periodic","selects a periodic boundary",meshes=badmesh)
badpde=copy.deepcopy(pde)
for item in badpde: item["ncu"]=2
bad=copy.deepcopy(interactions); bad[1]["fluxmap2"]=[1,2]
reject("width","one fint output width",pdes=badpde,edges=bad)

# Phase 2 selector-union partition contract. Face records may contain the same
# element more than once, but the partition contains each interface element
# exactly once. A one-entry vector must be identical to the legacy scalar.
pt = np.array([[0,1,2],[1,2,3]], dtype=np.int64)
pt2t = np.zeros((2,3), dtype=np.int64)
pf = np.array([[1,2,1],[2,3,4],[3,4,2],[4,1,3]], dtype=np.int64)
legacy = elementpartitionhdg([{}],pt,pt2t,1,0,None,pf,2)[0]
single = elementpartitionhdg([{}],pt,pt2t,1,0,None,pf,[2])[0]
union = elementpartitionhdg([{}],pt,pt2t,1,0,None,pf,[2,4])[0]
empty = elementpartitionhdg([{}],pt,pt2t,1,0,None,pf,[])[0]
assert np.array_equal(single['elempart'],legacy['elempart'])
assert np.array_equal(single['elempartpts'],legacy['elempartpts'])
assert np.array_equal(union['elempart'],np.array([0,1,2]))
assert np.array_equal(union['elempartpts'],np.array([0,3,0]))
assert np.array_equal(empty['elempart'],np.array([0,1,2]))
print("PASS: Python interaction validator rules and golden table")
