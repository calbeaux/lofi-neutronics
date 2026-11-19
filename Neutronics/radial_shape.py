"""Frozen, azimuthally averaged MPACT power and delayed-importance mapping."""
from types import SimpleNamespace
import h5py
import numpy as np
from scipy.optimize import brentq


def radial_profile(library, temperature, pitch):
    """Return annular radii, power/fuel-volume, importance, and outer support (SI)."""
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError('assembly_pitch_m must contain two positive finite lengths.')
    i, fraction = library._interval(temperature)
    key = (float(temperature), tuple(pitch))
    cache = getattr(library, '_radial_profiles', {})
    if key in cache:
        return cache[key]
    samples = []
    maps = []
    for sample in library._samples[i:i+2]:
        with h5py.File(sample['source']) as f:
            g = f[library.metadata['state']+'/NODAL_XS']
            mapping = f['CORE/computational_core_map'][()]
            vol = g['VOL'][()]*1e-6
            fuel = g['FUELVOLFRAC'][()]
            valid = vol > 0
            if fuel.shape != vol.shape or not np.isfinite(fuel[valid]).all() or np.any((fuel[valid]<-1e-12)|(fuel[valid]>1+1e-12)):
                raise ValueError('Invalid NODAL_XS/FUELVOLFRAC.')
            fuel = np.where(valid, np.clip(fuel,0,1), 0)
            arrays = []
            for name in ['KXSF', 'FLUX', 'ADJOINT FLUX', 'CHID']:
                a = g[name][()]
                if a.shape != (2,)+vol.shape or not np.isfinite(a[:,valid]).all() or np.any(a[:,valid]<0):
                    raise ValueError('Invalid radial field '+name)
                arrays.append(np.where(valid[None], a, 0))
            kf, phi, adj, chid = arrays
            power = np.sum(kf*100*phi, axis=0)
            importance = np.sum(adj*chid, axis=0)
            fueled_planes = np.any(power > 0, axis=(0,2))
            fv = vol*fuel*fueled_planes[None,:,None]
            samples.append((np.sum(vol*power,axis=(0,1)), np.sum(fv,axis=(0,1)),
                            np.sum(fv*importance,axis=(0,1))))
            maps.append(mapping)
    if not np.array_equal(*maps):
        raise ValueError('MPACT radial maps differ between temperature samples.')
    mapping = maps[0]
    count = len(samples[0][0])
    if mapping.ndim != 2 or not np.array_equal(np.sort(mapping[mapping>0]),np.arange(1,count+1)):
        raise ValueError('Computational core map must locate each nodal assembly exactly once.')
    values = [(1-fraction)*a+fraction*b for a,b in zip(*samples)]
    power, fuel, importance = values
    xy = np.empty((count,2))
    for row,col in zip(*np.nonzero(mapping)):
        xy[int(mapping[row,col])-1] = [(col+.5-mapping.shape[1]/2)*pitch[0],
                                      (mapping.shape[0]/2-row-.5)*pitch[1]]
    radius = np.linalg.norm(xy,axis=1)
    active = (fuel>0)&(power>0)
    if not np.any(active):
        raise ValueError('No fueled radial power samples.')
    bins = np.floor(radius/pitch.min()).astype(int)
    radii, heat, imp = [], [], []
    for b in np.unique(bins[active]):
        use = active&(bins==b)
        radii.append(np.average(radius[use],weights=fuel[use]))
        heat.append(power[use].sum()/fuel[use].sum())
        imp.append(importance[use].sum()/fuel[use].sum())
    support = (bins[active].max()+1)*pitch.min()
    result = (np.array(radii), np.array(heat), np.array(imp), float(support))
    cache[key] = result
    library._radial_profiles = cache
    return result


class RadialShape:
    def __init__(self, core, library, config):
        if library is None:
            raise ValueError('radial_shape requires an MPACT library.')
        if not isinstance(config,dict) or config.get('type') not in {'mpact_radius','cosine'}:
            raise ValueError('radial_shape.type must be mpact_radius or cosine.')
        if any(n.bounds is None for n in core.nodes):
            raise ValueError('Radial mapping requires channel bounds.')
        self.reference_temperature = float(config['reference_temperature'])
        self.profile = radial_profile(library,self.reference_temperature,config['assembly_pitch_m'])
        self.indices = np.array([[n.index for n in ch] for ch in core.channels])
        self.volumes = np.array([[n.volume for n in ch] for ch in core.channels])
        self.xy = np.array([(np.array(ch[0].bounds[:2])+np.array(ch[0].bounds[3:5]))/2 for ch in core.channels])
        center = (self.xy.min(axis=0)+self.xy.max(axis=0))/2
        self.radii = np.linalg.norm(self.xy-center,axis=1)
        r,p,j,support = self.profile
        if np.any(self.radii > support+1e-10):
            raise ValueError('Target channel radius extends beyond MPACT radial support.')
        raw = np.interp(self.radii,r,p)
        self.importance = np.interp(self.radii,r,j)[:,None]*np.ones_like(self.volumes)
        self.cosine_exponent = None
        if config['type'] == 'cosine':
            peak, edge = float(config.get('peak',1.3)), float(config.get('edge',.6))
            if not np.isfinite([peak,edge]).all() or not 0 < edge < 1 < peak:
                raise ValueError('Cosine heating requires 0 < edge < 1 < peak.')
            radius = float(self.radii.max())
            if radius <= 0:
                raise ValueError('Cosine heating requires channels at distinct radii.')
            scaled = np.clip(self.radii/radius,0,1)
            channel_volumes = self.volumes.sum(axis=1)

            def heating(exponent, x=scaled):
                return edge+(peak-edge)*np.cos(np.pi/2*x**exponent)

            def residual(exponent):
                return np.average(heating(exponent),weights=channel_volumes)-1

            if residual(.01)*residual(64.) > 0:
                raise ValueError('Requested cosine endpoints cannot give unit mean on this geometry.')
            self.cosine_exponent = brentq(residual,.01,64.,xtol=1e-12)
            raw = heating(self.cosine_exponent)
            plot_r = np.linspace(0,radius,201)
            self.profile = (plot_r, heating(self.cosine_exponent,plot_r/radius),
                            np.interp(plot_r,r,j),radius)
        denominator = (raw[:,None]*self.volumes).sum(axis=0)
        if np.any(denominator<=0) or np.any(self.importance.sum(axis=0)<=0):
            raise ValueError('Radial weights have zero normalization.')
        self.weights = raw[:,None]*self.volumes.sum(axis=0)/denominator
        self.core = core

    def broadcast(self, axial, target):
        target[self.indices] = np.asarray(axial)[None,:]*self.weights

    def collapse_xs(self, axial, temperatures, library, buckling):
        nodes = self.indices[:,axial]
        weights = self.weights[:,axial]*self.volumes[:,axial]
        weights /= weights.sum()
        values = [library.cross_sections(t,radial_buckling_squared=buckling(t)) for t in temperatures[nodes]]
        groups = []
        length = self.core.center_channel[axial].length
        for name in ['fast','th']:
            xs = [getattr(v,name) for v in values]
            fields = {key:float(np.dot(weights,[getattr(x,key) for x in xs])) for key in
                      ['a','s_out','nu_f','kappa_f','f','d_coef','radial_leakage']}
            fields['velocity'] = 1/np.dot(weights,[1/x.velocity for x in xs])
            boundary = np.dot(weights,[x.d_coef/(length/2+x.extrapolation_distance) for x in xs])
            fields['extrapolation_distance'] = fields['d_coef']/boundary-length/2
            groups.append(SimpleNamespace(**fields))
        return SimpleNamespace(fast=groups[0],th=groups[1])
