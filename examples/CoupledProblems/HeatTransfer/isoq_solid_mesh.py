"""Gmsh triangular solid meshes for the matching and non-matching ISOQ cases."""

import os, tempfile
import numpy as np
from exasim.Mesh.gmshcall import gmshcall
from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.getelemface import getelemface
from exasim.Preprocessing.masternodes import masternodes
from exasim.Preprocessing.mkshape import mkshape


def _ordered_wall(fluid, boundary=4):
    face=getelemface(2,1); lf,elem=np.nonzero(fluid["f"]==boundary)
    edges=np.column_stack([fluid["t"][face[:,f],e] for f,e in zip(lf,elem)]).T
    adjacency={}
    for i,(a,b) in enumerate(edges):
        adjacency.setdefault(int(a),[]).append(i); adjacency.setdefault(int(b),[]).append(i)
    ends=[node for node,links in adjacency.items() if len(links)==1]
    current=min(ends,key=lambda node:(fluid["p"][1,node],fluid["p"][0,node]))
    used=set(); chain=[current]; ordered_e=[]; ordered_f=[]
    for _ in range(len(edges)):
        candidates=[i for i in adjacency[current] if i not in used]
        if len(candidates)!=1: raise RuntimeError("fluid wall is not one open edge chain")
        i=candidates[0]; used.add(i); ordered_e.append(int(elem[i])); ordered_f.append(int(lf[i]))
        current=int(edges[i,1] if edges[i,0]==current else edges[i,0]); chain.append(current)
    return fluid["p"][:,chain].T,np.array(ordered_e),np.array(ordered_f)


def _write_geo(path,wall,ymin,xmax,h,matching):
    with open(path,"w",encoding="ascii") as stream:
        for i,(x,y) in enumerate(wall,1): stream.write(f"Point({i}) = {{{x:.17g}, {y:.17g}, 0, {h:.17g}}};\n")
        corner=len(wall)+1; stream.write(f"Point({corner}) = {{{xmax:.17g}, {ymin:.17g}, 0, {h:.17g}}};\n")
        if matching:
            for i in range(1,len(wall)): stream.write(f"Line({i}) = {{{i}, {i+1}}};\n")
            stream.write(f"Line({len(wall)}) = {{{len(wall)}, {corner}}};\nLine({len(wall)+1}) = {{{corner}, 1}};\n")
            ids=", ".join(str(i) for i in range(1,len(wall)+2)); stream.write(f"Curve Loop(1) = {{{ids}}};\nPlane Surface(1) = {{1}};\n")
            stream.write("Transfinite Curve {"+", ".join(str(i) for i in range(1,len(wall)))+"} = 2;\n")
            stream.write(f"Mesh.MeshSizeMin = {0.25*h:.17g};\nMesh.MeshSizeMax = {h:.17g};\n")
        else:
            ids=", ".join(str(i) for i in range(1,len(wall)+1)); stream.write(f"Spline(1) = {{{ids}}};\n")
            stream.write(f"Line(2) = {{{len(wall)}, {corner}}};\nLine(3) = {{{corner}, 1}};\nCurve Loop(1) = {{1,2,3}};\nPlane Surface(1) = {{1}};\n")
            stream.write(f"Mesh.MeshSizeMin = {0.95*h:.17g};\nMesh.MeshSizeMax = {1.05*h:.17g};\nMesh.MeshSizeFromPoints = 1;\nMesh.MeshSizeFromCurvature = 0;\nMesh.MeshSizeExtendFromBoundary = 0;\nMesh.Smoothing = 20;\nMesh.Optimize = 1;\nMesh.OptimizeNetgen = 1;\n")
        stream.write("Mesh.Algorithm = 6;\nMesh.ElementOrder = 1;\nMesh.RecombineAll = 0;\n")


def _golden_minimum(function):
    a,b=0.0,1.0; ratio=(np.sqrt(5.0)-1.0)/2.0; c=b-ratio*(b-a); d=a+ratio*(b-a); fc,fd=function(c),function(d)
    for _ in range(70):
        if fc<fd: b,d,fd=d,c,fc; c=b-ratio*(b-a); fc=function(c)
        else: a,c,fc=c,d,fd; d=a+ratio*(b-a); fd=function(d)
    x=0.5*(a+b); return x,function(x)


def _project(points,fluid,wall,elements,faces,porder):
    plocfc=masternodes(porder,1,1)[0]; perm=np.asarray(masternodes(porder,2,1)[4],dtype=int)-1
    face_nodes=[]
    for i,(e,f) in enumerate(zip(elements,faces)):
        nodes=fluid["dgnodes"][perm[:,f],:,e]
        if np.linalg.norm(nodes[0]-wall[i])+np.linalg.norm(nodes[-1]-wall[i+1]) > np.linalg.norm(nodes[0]-wall[i+1])+np.linalg.norm(nodes[-1]-wall[i]): nodes=nodes[::-1]
        face_nodes.append(nodes)
    def evaluate(nodes,xi): return mkshape(porder,plocfc,np.array([[xi]]),1)[:,:,0].T@nodes
    result=np.empty_like(points)
    for j,point in enumerate(points):
        best=(np.inf,point)
        for nodes in face_nodes:
            objective=lambda xi: float(np.sum((evaluate(nodes,xi)[0]-point)**2))
            xi,value=_golden_minimum(objective)
            for candidate in (0.0,xi,1.0):
                mapped=evaluate(nodes,candidate)[0]; dist=np.sum((mapped-point)**2)
                if dist<best[0]: best=(dist,mapped)
        result[j]=best[1]
    return result


def make_isoq_solid(pde,fluid,matching,mesh_size=0.0028):
    wall,elements,faces=_ordered_wall(fluid); ymin=np.min(fluid["p"][1]); xmax=np.max(fluid["p"][0]); scale=max(1.0,np.ptp(fluid["p"],axis=1).max()); tol=1e-10*scale
    with tempfile.TemporaryDirectory(prefix="isoqsolid_") as directory:
        base=os.path.join(directory,"solid"); _write_geo(base+".geo",wall,ymin,xmax,mesh_size,matching); p,t=gmshcall(pde,base,2,0)
    area=(p[0,t[1]]-p[0,t[0]])*(p[1,t[2]]-p[1,t[0]])-(p[1,t[1]]-p[1,t[0]])*(p[0,t[2]]-p[0,t[0]])
    flip=area<0
    if np.any(flip):
        second=t[1,flip].copy(); t[1,flip]=t[2,flip]; t[2,flip]=second
    startx,endy=wall[0,0],wall[-1,1]
    if matching:
        wall_vertices=[]
        for point in wall:
            distance=np.linalg.norm(p.T-point,axis=1); node=int(np.argmin(distance));
            if distance[node]>tol: raise RuntimeError("Gmsh did not preserve a fluid wall vertex")
            p[:,node]=point; wall_vertices.append(node)
        wall_vertices=np.array(wall_vertices)
        onwall=lambda z: np.min(np.linalg.norm(z.T[:,None,:]-wall[None,:,:],axis=2),axis=1)<=tol
    else:
        onwall=lambda z: ~((np.abs(z[1]-ymin)<tol)&(z[0]>startx+tol) | (np.abs(z[0]-xmax)<tol)&(z[1]<endy-tol))
    boundaryexpr=[onwall,lambda z:np.abs(z[1]-ymin)<tol,lambda z:np.abs(z[0]-xmax)<tol]
    f=facenumbering(p,t,0,boundaryexpr,[])[0]
    if not matching:
        vertices=np.unique(np.concatenate([t[getelemface(2,0)[:,lf],e] for lf,e in zip(*np.nonzero(f==1))])); p[:,vertices]=_project(p[:,vertices].T,fluid,wall,elements,faces,pde["porder"]).T
        f=facenumbering(p,t,0,boundaryexpr,[])[0]
    dgnodes=createdgnodes(p,t,f,[],[],pde["porder"]); perm=np.asarray(masternodes(pde["porder"],2,0)[4],dtype=int)-1
    lf_s,e_s=np.nonzero(f==1)
    if matching:
        edge=getelemface(2,0)
        for lf,e in zip(lf_s,e_s):
            vertices=t[edge[:,lf],e]; segment=next(i for i in range(len(wall_vertices)-1) if set(vertices)==set(wall_vertices[i:i+2]))
            target=fluid["dgnodes"][np.asarray(masternodes(pde["porder"],2,1)[4],dtype=int)[...,faces[segment]]-1,:,elements[segment]]
            if vertices[0]==wall_vertices[segment+1]: target=target[::-1]
            dgnodes[perm[:,lf],:,e]=target
    else:
        for lf,e in zip(lf_s,e_s): dgnodes[perm[:,lf],:,e]=_project(dgnodes[perm[:,lf],:,e],fluid,wall,elements,faces,pde["porder"])
    return {"p":p,"t":t,"f":f,"dgnodes":dgnodes,"udg":[],"vdg":[],"wdg":[],"tprd":[],"boundaryexpr":boundaryexpr,"boundarycondition":np.array([3,2,1]),"interfacecondition":np.array([1,0,0]),"periodicexpr":[],"curvedboundary":np.array([],dtype=int),"curvedboundaryexpr":[],"periodicboundary":np.array([],dtype=int)}
