"""HSSM adapters for ddmsa_logp. t is the lower edge of the non-decision
distribution (t0 = t + st/2). HSSM's own DDM takes a as the half separation,
so the cavanagh adapter doubles a and sa; the ITC adapters work in full units."""

import pytensor.tensor as pt

from saddm import ddmsa_logp


def _split(data):
    data = pt.reshape(data, (-1, 2))
    return pt.abs(data[:, 0]), data[:, 1]


def ddmsa_half_a(data, v, a, z, t, sv, sa, st):
    rt, ch = _split(data)
    return ddmsa_logp(rt, ch, a=2.0 * a, z=z, v=v, t=t + st / 2.0, sv=sv, sa=2.0 * sa, st=st)


def ddmsa(data, v, a, z, t, sv, sa, st):
    rt, ch = _split(data)
    return ddmsa_logp(rt, ch, a=a, z=z, v=v, t=t + st / 2.0, sv=sv, sa=sa, st=st)


def ddm(data, v, a, z, t):
    rt, ch = _split(data)
    return ddmsa_logp(rt, ch, a=a, z=z, v=v, t=t)
