"""Two-block ISOQ mesh shared by the Python verification driver."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Mesh.gmshcall import gmshcall
from exasim.Preprocessing.getelemface import getelemface
from exasim.Preprocessing.masternodes import masternodes
from exasim.Preprocessing.mkshape import mkshape


def _loginc(values, alpha):
    lower, upper = np.min(values), np.max(values)
    if upper == lower:
        return np.array(values, copy=True)
    return lower + (upper - lower) * (
        np.exp(alpha * (values - lower) / (upper - lower)) - 1.0
    ) / (np.exp(alpha) - 1.0)


def _logdec(values, alpha):
    lower, upper = np.min(values), np.max(values)
    if upper == lower:
        return np.array(values, copy=True)
    return lower + (upper - lower) * (
        1.0 - np.exp(-alpha * (values - lower) / (upper - lower))
    ) / (1.0 - np.exp(-alpha))


def _rectangular_quad_mesh(nx, ny, porder):
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    points = np.vstack((x, y))
    elements = np.empty((4, nx * ny), dtype=np.int64)
    element = 0
    for iy in range(ny):
        for ix in range(nx):
            lower_left = ix + iy * (nx + 1)
            elements[:, element] = [
                lower_left,
                lower_left + 1,
                lower_left + nx + 2,
                lower_left + nx + 1,
            ]
            element += 1
    dgnodes = createdgnodes(
        points, elements, np.zeros((4, elements.shape[1]), dtype=np.int64),
        [], [], porder,
    )
    return points, elements, dgnodes


def _inverse_bilinear(point, coefficients, initial):
    xi, eta = initial
    for _ in range(100):
        basis = np.array([1.0, xi, eta, xi * eta])
        residual = basis @ coefficients - point
        jacobian = np.array(
            [
                [coefficients[1, 0] + coefficients[3, 0] * eta,
                 coefficients[2, 0] + coefficients[3, 0] * xi],
                [coefficients[1, 1] + coefficients[3, 1] * eta,
                 coefficients[2, 1] + coefficients[3, 1] * xi],
            ]
        )
        update = np.linalg.solve(jacobian, -residual)
        xi += update[0]
        eta += update[1]
        if np.linalg.norm(update) < 1.0e-10:
            return np.array([xi, eta])
    raise RuntimeError("inverse bilinear map did not converge")


def _forward_bilinear(reference_points, coefficients):
    xi, eta = reference_points[:, 0], reference_points[:, 1]
    basis = np.column_stack((np.ones(xi.size), xi, eta, xi * eta))
    return basis @ coefficients


def _surfmesh2d(
    lower_curve, upper_curve, nx, ny, porder,
    streamwise_scaling=(2.0, 1.5), normal_scaling=(3.0, 1.0e-8),
):
    matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    corners = np.vstack(
        (lower_curve[0], lower_curve[-1], upper_curve[-1], upper_curve[0])
    )
    coefficients = np.linalg.solve(matrix, corners)

    lower_reference = np.vstack(
        [
            _inverse_bilinear(point, coefficients, (i / (len(lower_curve) - 1), 0.0))
            for i, point in enumerate(lower_curve)
        ]
    )
    upper_reference = np.vstack(
        [
            _inverse_bilinear(point, coefficients, (i / (len(upper_curve) - 1), 1.0))
            for i, point in enumerate(upper_curve)
        ]
    )
    lower_fit = np.polyfit(lower_reference[:, 0], lower_reference[:, 1],
                           min(len(lower_curve) - 1, 12))
    upper_fit = np.polyfit(upper_reference[:, 0], upper_reference[:, 1],
                           min(len(upper_curve) - 1, 12))

    points, elements, dgnodes = _rectangular_quad_mesh(nx, ny, porder)
    vertex_reference = points.T.copy()
    node_reference = np.column_stack(
        (
            dgnodes[:, 0, :].reshape(-1, order="F"),
            dgnodes[:, 1, :].reshape(-1, order="F"),
        )
    )
    sides = {
        "vertex_left": np.abs(vertex_reference[:, 0]) < 1.0e-6,
        "vertex_right": np.abs(vertex_reference[:, 0] - 1.0) < 1.0e-6,
        "node_left": np.abs(node_reference[:, 0]) < 1.0e-6,
        "node_right": np.abs(node_reference[:, 0] - 1.0) < 1.0e-6,
    }

    for reference in (vertex_reference, node_reference):
        reference[:, 0] = _logdec(
            _loginc(reference[:, 0], streamwise_scaling[0]),
            max(streamwise_scaling[1], 1.0e-8),
        )
        reference[:, 1] = _logdec(
            _loginc(reference[:, 1], normal_scaling[0]),
            max(normal_scaling[1], 1.0e-8),
        )
        lower = np.polyval(lower_fit, reference[:, 0])
        upper = np.polyval(upper_fit, reference[:, 0])
        endpoints = (np.abs(reference[:, 0]) < 1.0e-6) | (
            np.abs(reference[:, 0] - 1.0) < 1.0e-6
        )
        lower[endpoints] = 0.0
        upper[endpoints] = 1.0
        reference[:, 1] = lower + (upper - lower) * reference[:, 1]

    physical_points = _forward_bilinear(vertex_reference, coefficients)
    physical_nodes = _forward_bilinear(node_reference, coefficients)
    dgnodes[:, 0, :] = physical_nodes[:, 0].reshape(
        dgnodes.shape[0], dgnodes.shape[2], order="F"
    )
    dgnodes[:, 1, :] = physical_nodes[:, 1].reshape(
        dgnodes.shape[0], dgnodes.shape[2], order="F"
    )
    return physical_points.T, elements, dgnodes, sides


def _isoq_curves():
    nose_radius = 0.102
    shoulder_radius = nose_radius / 16.0
    end_x = 0.06
    nose_angle = 27.82 * np.pi / 180.0

    theta = np.linspace(0.0, nose_angle, 481)
    nose = np.column_stack(
        (nose_radius * (1.0 - np.cos(theta)), nose_radius * np.sin(theta))
    )
    initial = np.pi - nose_angle
    center = nose[-1] - shoulder_radius * np.array([np.cos(initial), np.sin(initial)])
    theta = np.linspace(initial, np.pi / 2.0, 49)
    shoulder = center + shoulder_radius * np.column_stack((np.cos(theta), np.sin(theta)))
    afterbody = np.column_stack(
        (np.linspace(shoulder[-1, 0], end_x, 240),
         np.full(240, shoulder[-1, 1]))
    )
    lower = np.vstack((nose[:-1], shoulder[:-1], afterbody))

    theta = np.linspace(np.pi, np.pi / 2.0, 2000)
    upper = np.column_stack(
        (3.7 * end_x + 4.4 * end_x * np.cos(theta),
         3.6 * end_x * np.sin(theta))
    )
    upper = upper[upper[:, 0] <= end_x]
    upper[-1, 0] = end_x
    return lower, upper


def _connect_blocks(first, second, tolerance=1.0e-5):
    p1, t1, dgnodes1, side1 = first
    p2, t2, dgnodes2, side2 = second
    average_vertices = 0.5 * (
        p1[:, side1["vertex_right"]].T + p2[:, side2["vertex_left"]].T
    )
    p1[:, side1["vertex_right"]] = average_vertices.T
    p2[:, side2["vertex_left"]] = average_vertices.T

    flat1 = np.column_stack(
        (dgnodes1[:, 0, :].reshape(-1, order="F"),
         dgnodes1[:, 1, :].reshape(-1, order="F"))
    )
    flat2 = np.column_stack(
        (dgnodes2[:, 0, :].reshape(-1, order="F"),
         dgnodes2[:, 1, :].reshape(-1, order="F"))
    )
    average_nodes = 0.5 * (
        flat1[side1["node_right"]] + flat2[side2["node_left"]]
    )
    flat1[side1["node_right"]] = average_nodes
    flat2[side2["node_left"]] = average_nodes
    for flat, nodes in ((flat1, dgnodes1), (flat2, dgnodes2)):
        nodes[:, 0, :] = flat[:, 0].reshape(nodes.shape[0], nodes.shape[2], order="F")
        nodes[:, 1, :] = flat[:, 1].reshape(nodes.shape[0], nodes.shape[2], order="F")

    mapping = np.empty(p2.shape[1], dtype=np.int64)
    keep = np.ones(p2.shape[1], dtype=bool)
    next_index = p1.shape[1]
    for index in range(p2.shape[1]):
        squared_distance = np.sum((p1.T - p2[:, index]) ** 2, axis=1)
        nearest = int(np.argmin(squared_distance))
        if squared_distance[nearest] < tolerance**2:
            mapping[index] = nearest
            keep[index] = False
        else:
            mapping[index] = next_index
            next_index += 1
    points = np.hstack((p1, p2[:, keep]))
    elements = np.hstack((t1, mapping[t2]))
    dgnodes = np.concatenate((dgnodes1, dgnodes2), axis=2)
    return points, elements, dgnodes


def make_isoq_mesh(porder, radial_shift=5.0e-4):
    lower, upper = _isoq_curves()
    upper_first = upper[upper[:, 0] <= -0.02]
    upper_second = np.vstack((upper_first[-1], upper[upper[:, 0] > -0.02]))
    lower_first = lower[lower[:, 0] <= 0.013]
    lower_second = np.vstack((lower_first[-1], lower[lower[:, 0] > 0.013]))

    first = _surfmesh2d(lower_first, upper_first, 30, 60, porder)
    second = _surfmesh2d(lower_second, upper_second, 25, 60, porder)
    points, elements, dgnodes = _connect_blocks(first, second)
    points[1, :] += radial_shift
    dgnodes[:, 1, :] += radial_shift

    lower_y = np.min(points[1, :])
    right_x = np.max(points[0, :])
    boundaryexpr = [
        lambda p: np.abs(p[1, :] - lower_y) < 1.0e-6,
        lambda p: p[0, :] > right_x - 1.0e-4,
        lambda p: (p[0, :] < -1.0e-3) | (p[1, :] > 0.1),
        lambda p: np.abs(p[0, :]) < 20.0 + 1.0e-6,
    ]
    faces = facenumbering(points, elements, 1, boundaryexpr, [])[0]
    return {
        "p": points,
        "t": elements,
        "f": faces,
        "dgnodes": dgnodes,
        "udg": [],
        "vdg": [],
        "wdg": [],
        "tprd": [],
        "boundaryexpr": boundaryexpr,
        "boundarycondition": np.array([5, 2, 1, 3], dtype=np.int64),
        "periodicexpr": [],
        "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [],
        "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder):
    perm = np.asarray(masternodes(porder, 2, 1)[4], dtype=np.int64) - 1
    local_faces, elements = np.nonzero(mesh["f"] == 4)
    wall_nodes = np.concatenate(
        [mesh["dgnodes"][perm[:, face], :, elem]
         for face, elem in zip(local_faces, elements)], axis=0
    )
    npe, _, ne = mesh["dgnodes"].shape
    distance = np.empty((npe, 1, ne), dtype=np.float64, order="F")
    for elem in range(ne):
        delta = mesh["dgnodes"][:, :, elem, None] - wall_nodes.T[None, :, :]
        distance[:, 0, elem] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance


# Gmsh triangular solid meshes for the matching and non-matching ISOQ cases.


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
