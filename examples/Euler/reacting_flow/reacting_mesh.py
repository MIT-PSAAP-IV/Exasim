"""Curved half-cylinder mesh helpers for the reacting-flow example."""

import numpy as np
from exasim.Preprocessing.createdgnodes import createdgnodes
from exasim.Preprocessing.facenumbering import facenumbering
from exasim.Preprocessing.masternodes import masternodes


def _logdec(values, alpha):
    lo, hi = np.min(values), np.max(values)
    return lo + (hi-lo) * (1.0-np.exp(-alpha*(values-lo)/(hi-lo))) / (1.0-np.exp(-alpha))


def _quad_mesh(nx, ny):
    x = np.tile(np.linspace(0.,1.,nx+1),ny+1); y = np.repeat(np.linspace(0.,1.,ny+1),nx+1)
    p = np.vstack((x,y)); cells=[]
    for j in range(ny):
        for i in range(nx):
            n=i+j*(nx+1); cells.append([n,n+1,n+nx+2,n+nx+1])
    return p,np.asarray(cells,dtype=np.int64).T


def _map(points):
    theta=1.5*np.pi + points[1,...]*(0.5*np.pi-1.5*np.pi)
    outer=-1.25*np.cos(theta)+2.25*(1.0+np.cos(theta))
    radius=outer+points[0,...]*(1.0-outer)
    return np.stack((radius*np.cos(theta),radius*np.sin(theta)),axis=0)


def make_mesh(porder):
    p,t=_quad_mesh(80,80); dgnodes=createdgnodes(p,t,np.zeros((4,t.shape[1]),dtype=np.int64),[],[],porder)
    p[0,:]=_logdec(p[0,:],0.5); dgnodes[:,0,:]=_logdec(dgnodes[:,0,:],0.5)
    p=_map(p); dgnodes=np.transpose(_map(np.transpose(dgnodes,(1,0,2))),(1,0,2))
    boundaryexpr=[lambda x: np.sqrt(x[0,:]**2+x[1,:]**2)<1.+1.e-6,lambda x:x[0,:]>-1.e-7,lambda x:np.abs(x[0,:])<20.]
    f=facenumbering(p,t,1,boundaryexpr,[])[0]
    return {"p":p,"t":t,"f":f,"dgnodes":dgnodes,"udg":[],"vdg":[],"wdg":[],"tprd":[],"boundaryexpr":boundaryexpr,"boundarycondition":np.array([3,6,5]),"periodicexpr":[],"curvedboundary":np.array([],dtype=int),"curvedboundaryexpr":[],"periodicboundary":np.array([],dtype=int)}


def wall_distance(mesh,porder):
    perm=np.asarray(masternodes(porder,2,1)[4],dtype=int)-1
    faces,elements=np.nonzero(mesh["f"]==1)
    wall=np.concatenate([mesh["dgnodes"][perm[:,face],:,element] for face,element in zip(faces,elements)])
    npe,_,ne=mesh["dgnodes"].shape; distance=np.empty((npe,1,ne),order="F")
    for element in range(ne):
        delta=mesh["dgnodes"][:,:,element,None]-wall.T[None,:,:]
        distance[:,0,element]=np.sqrt(np.sum(delta*delta,axis=1)).min(axis=1)
    return distance


def element_scales(mesh):
    npe,_,ne=mesh["dgnodes"].shape; scales=np.empty((npe,ne))
    for element in range(ne):
        vertices=mesh["p"][:,mesh["t"][:,element]]
        dx=vertices[:,1]-vertices[:,0]; dy=vertices[:,3]-vertices[:,0]
        scales[:,element]=np.sqrt(abs(dx[0]*dy[1]-dx[1]*dy[0]))
    return scales
