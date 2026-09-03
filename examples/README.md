# HSSM examples

| script | what it fits |
|---|---|
| `estimate_HSSM_saddm.py` | Flat DDM-SA on `cavanagh_theta` (ships with HSSM) — the minimal adapter for registering `ddmsa_logp` as a custom analytical likelihood. |
| `HSSM_estimate_ITC_Amasino.py` | Per-subject fits (212 subjects; `PLAIN=1` for the plain-DDM benchmark spec) and the population k-sweep on the exact permutation files the Fortran campaign used. |
| `HSSM_hierarchical.py` | Hierarchical variants (`VARIANT` env var): cavanagh DDM-SA with participant random effects; ITC plain and full-SA hierarchies. |

## Conventions the adapters encode

- **HSSM's `a` is the half boundary separation** (its built-in DDM doubles it
  internally); `saddm` uses the full separation, so cavanagh adapters pass
  `2*a` and `2*sa`. The ITC scripts define `a` in full-separation units
  directly, matching the Fortran benchmark.
- **Non-decision time is sampled by its lower edge.** With `st > 0` the
  earliest response is at `t − st/2`, so bounding `t` by min(RT) excludes the
  truth (and spuriously collapses `sa`); sampling `t` and `st` independently
  gives NUTS an untraversable ridge. Adapters therefore pass `t + st/2` to the
  likelihood and bound the sampled `t` (the edge) by min(RT); report
  `t0 = t + st/2`.
- **Hierarchical random effects on `a`/`t` need bounded links**
  (`link_settings="log_logit"`). Identity-link offsets push parameters into
  the invalid region and every proposal diverges.
- Set HSSM bounds so the midpoint (HSSM's default initial value) is feasible —
  e.g. a `t` bound of (0, 2) starts sampling at 1.0, above most min RTs.

## Environment

Use the `hssm` extra (`pip install -e ".[hssm]"`). The jax pin is deliberate:
jax ≥ 0.10 silently freezes numpyro NUTS at its initial point;
`jax==0.5.3` / `numpyro==0.19.0` are verified. For parallel chains call
`numpyro.set_host_device_count(n)` before jax initializes (the scripts do).
