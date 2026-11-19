"""Radial mapping and all-channel feedback, without the native backend."""
import h5py
import numpy as np
import pytest
from Coupling.core_geometry import CoreGeometry
from Coupling.state import State
from Neutronics.radial_shape import RadialShape
from Neutronics.mpact_library import MPACTLibrary
from Neutronics.point_kinetics import PointKineticsStrategy
from Neutronics.diffusion import Diffusion
from test_mpact_library import make_sample
from test_multichannel import mesh, wire

CONFIG = dict(type='mpact_radius',reference_temperature=900.,assembly_pitch_m=[1.,1.])

@pytest.fixture
def library(tmp_path):
    paths=[]
    for name,t in [('cold',800),('hot',1000)]:
        path=make_sample(tmp_path/(name+'.h5'),t,scale=1 if t==800 else 1.1)
        with h5py.File(path,'a') as f:
            f['CORE/computational_core_map']=np.arange(1,26).reshape(5,5)
            g=f['STATE_0001/NODAL_XS']
            for k in list(g):
                a=g[k][()]
                if a.ndim>=3:
                    del g[k];g[k]=np.repeat(a,25,axis=-1)
            g['FUELVOLFRAC']=np.ones(g['VOL'].shape)
            radius=np.linalg.norm(np.indices((5,5)).reshape(2,-1).T-2,axis=1)
            phi=g['FLUX'][()];phi*= (3-.5*radius)[None,None,None,:]
            del g['FLUX'];g['FLUX']=phi
        paths.append(path)
    return MPACTLibrary.from_files(paths)


def test_mapping_symmetry_conservation_and_cache(library,monkeypatch):
    core=CoreGeometry(mesh(volumes=(1,2,3)))
    shape=RadialShape(core,library,CONFIG)
    np.testing.assert_allclose(shape.weights[0],shape.weights[2])
    assert np.all(shape.weights[1]>shape.weights[0])
    np.testing.assert_allclose((shape.weights*shape.volumes).sum(axis=0),core.axial_volumes)
    target=np.zeros(100)
    axial=np.array([1.,2.,3.,4.])
    core.broadcast(axial,target,shape)
    np.testing.assert_allclose((target[shape.indices]*shape.volumes).sum(axis=0),axial*core.axial_volumes)
    assert target[0]==0
    monkeypatch.setattr(h5py,'File',lambda *a,**kw:pytest.fail('cached profile reopened a file'))
    RadialShape(core,library,CONFIG)


@pytest.mark.parametrize('strategy',['pk','diffusion'])
def test_outer_channel_feedback_and_power_conservation(library,strategy):
    cls=PointKineticsStrategy if strategy=='pk' else Diffusion
    solver=cls(mesh(),initial_power=1e6,library=library,radial_shape=CONFIG)
    data=wire(solver)
    solver.set_solver_method(State('steadystate'));solver.update_power()
    assert sum(data['power'][n.index]*n.volume for n in solver.core.nodes)==pytest.approx(1e6)
    r=solver.radial
    oldT=solver.get_core_temperature()
    oldsource=solver.get_core_precursor_sum() if strategy=='pk' else solver.total_precursor_sum()
    before=solver.get_xs(solver._axial_nodes[0]).fast.a if strategy=='diffusion' else 0
    data['temperature'][r.indices[0,0]]+=20
    data['precursor_source'][r.indices[0,0]]+=1e10
    assert solver.get_core_temperature()>oldT
    newsource=solver.get_core_precursor_sum() if strategy=='pk' else solver.total_precursor_sum()
    assert newsource>oldsource
    if strategy=='diffusion':
        assert solver.get_xs(solver._axial_nodes[0]).fast.a!=before
    else:
        solver.set_solver_method(State('steadystate'));solver.update_power()
        weights=r.weights.copy();pop=solver._n_0
        solver.set_solver_method(State(dict(time_step=.05,time_duration=1,scheme='BDF1')))
        solver.update_power()
        assert solver.amplitude==pytest.approx(1)
        assert solver._n_0==pop
        np.testing.assert_array_equal(r.weights,weights)


def test_diffusion_uniform_projection_and_local_buckling(library):
    radial=Diffusion(mesh(volumes=(1,2,3)),1e6,library=library,radial_shape=CONFIG)
    raw=Diffusion(mesh(volumes=(1,2,3)),1e6,library=library)
    a,b=wire(radial),wire(raw)
    for x,y in zip(radial.set_up_matrix(),raw.set_up_matrix()):
        np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
    r=radial.radial;a['temperature'][r.indices[:,0]]=[820,900,980]
    radial.buckling_squared=lambda t:.01*t
    xs=radial.get_xs(radial._axial_nodes[0])
    weights=r.weights[:,0]*r.volumes[:,0];weights/=weights.sum()
    local=[library.cross_sections(t,radial_buckling_squared=.01*t).fast for t in [820,900,980]]
    assert xs.fast.radial_leakage==pytest.approx(sum(w*x.radial_leakage for w,x in zip(weights,local)))
    assert 1/xs.fast.velocity==pytest.approx(sum(w/x.velocity for w,x in zip(weights,local)))


def test_missing_data_bounds_and_support(library):
    from test_mpact_library import diffusion_nodes
    with pytest.raises(ValueError,match='requires an MPACT library'):
        RadialShape(CoreGeometry(mesh()),None,CONFIG)
    with pytest.raises(ValueError,match='channel bounds'):
        RadialShape(CoreGeometry(diffusion_nodes()),library,CONFIG)
    config={**CONFIG,'assembly_pitch_m':[.01,.01]}
    with pytest.raises(ValueError,match='beyond MPACT'):
        RadialShape(CoreGeometry(mesh()),library,config)
    with h5py.File(library._samples[0]['source'],'a') as f:
        del f['STATE_0001/NODAL_XS/FUELVOLFRAC']
    with pytest.raises(KeyError):
        RadialShape(CoreGeometry(mesh()),library,CONFIG)


def test_uniform_radial_weights_reproduce_pk_and_source_projection(library,monkeypatch):
    import Neutronics.radial_shape as module
    monkeypatch.setattr(module,'radial_profile',lambda *args:(np.array([0.,3.]),np.ones(2),np.ones(2),3.))
    old=PointKineticsStrategy(mesh(),initial_power=1e6,library=library)
    new=PointKineticsStrategy(mesh(),initial_power=1e6,library=library,radial_shape=CONFIG)
    a,b=wire(old),wire(new)
    for data in [a,b]:data['precursor_source'][:]=1e10
    for solver in [old,new]:
        solver.set_solver_method(State('steadystate'));solver.update_power()
    np.testing.assert_allclose(a['power'],b['power'])
    assert old.get_core_precursor_sum()==pytest.approx(new.get_core_precursor_sum())
    for solver,data in [(old,a),(new,b)]:
        data['temperature'][:]=920
        solver.set_solver_method(State(dict(time_step=.05,time_duration=1,scheme='BDF1')))
        solver.update_power()
    assert old.amplitude==pytest.approx(new.amplitude)
    d=Diffusion(mesh(volumes=(1,2,3)),1e6,library=library,radial_shape=CONFIG)
    data=wire(d);r=d.radial
    data['precursor_source'][r.indices]=np.arange(r.indices.size).reshape(r.indices.shape)+1
    for i,node in enumerate(d._axial_nodes):
        density=data['precursor_source'][r.indices[:,i]].sum()/r.volumes[:,i].sum()
        assert d.projected_precursor_source(node)/node.volume==pytest.approx(density)


def test_geometry_change_remaps_radial_weights(library):
    d=Diffusion(mesh(),1e6,library=library,radial_shape=CONFIG)
    previous=d.radial
    d.set_core_nodes(mesh(volumes=(3,1,1),offset=(1.,2.,0.)))
    assert d.radial is not previous
    np.testing.assert_allclose(d.radial.radii,previous.radii)
    np.testing.assert_allclose((d.radial.weights*d.radial.volumes).sum(axis=0),d.core.axial_volumes)


def test_cosine_endpoints_monotonicity_and_conservation(library):
    config={**CONFIG,'type':'cosine','peak':1.3,'edge':.6}
    # Three radii are needed to fit the mean without changing either endpoint.
    from Coupling.node import Node
    nodes=[]
    for c,x in enumerate([-2.,-1.,0.,1.,2.]):
        for z in range(4):
            nodes.append(Node(.2,.01,c*4+z,tuple([x-.1,-.1,z*.2,x+.1,.1,(z+1)*.2])))
    core=CoreGeometry(nodes)
    r=RadialShape(core,library,config)
    assert r.weights[2,0]==pytest.approx(1.3)
    assert r.weights[0,0]==pytest.approx(.6)
    assert r.weights[-1,0]==pytest.approx(.6)
    assert r.weights[:,0].mean()==pytest.approx(1)
    assert np.all(np.diff(r.profile[1])<=0)
    target=np.zeros(20);core.broadcast(np.array([1.,2.,3.,4.]),target,r)
    np.testing.assert_allclose((target[r.indices]*r.volumes).sum(axis=0),core.axial_volumes*np.arange(1,5))
    mpact=RadialShape(core,library,CONFIG)
    np.testing.assert_allclose(r.importance,mpact.importance)
    changed=RadialShape(core,library,{**config,'peak':1.4})
    assert changed.weights[2,0]==pytest.approx(1.4)
    with pytest.raises(ValueError,match='0 < edge < 1 < peak'):
        RadialShape(core,library,{**config,'edge':1.1})
