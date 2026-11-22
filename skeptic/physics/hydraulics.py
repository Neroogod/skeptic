"""Drilling hydraulics (field units: in, ft, gpm, ppg, psi, cP, lb/100 ft^2).
Equations: Bourgoyne et al., Applied Drilling Engineering (1986), ch. 4."""
import math
import re


def pump_output(liner_in, stroke_in, spm, efficiency_pct=95.0, cylinders=3):
    """Single-acting triplex: displacement per stroke = cylinders * pi/4 * D^2 * L (in^3).
    Returns gal/stroke and gpm at the given volumetric efficiency."""
    gal_per_stroke = cylinders * math.pi / 4 * liner_in ** 2 * stroke_in / 231.0 * efficiency_pct / 100.0
    return dict(gal_per_stroke=round(gal_per_stroke, 3), bbl_per_stroke=round(gal_per_stroke / 42.0, 4),
                gpm=round(gal_per_stroke * spm, 1))


def tfa_from_nozzles(nozzles):
    """'4x12, 4x11' or '9x13' or '1X24 3X20' (sizes in 32nds of an inch) -> total flow area (in^2)."""
    pairs = re.findall(r"(\d+)\s*[xX]\s*(\d+)", str(nozzles))
    if not pairs:
        raise ValueError(f"no nozzle sizes in {nozzles!r} (expected e.g. '3x20, 1x24')")
    return round(sum(int(n) * math.pi / 4 * (int(d) / 32.0) ** 2 for n, d in pairs), 4)


def bit_pressure_drop(mw_ppg, q_gpm, tfa_in2, cd=0.95):
    """Pressure drop across the bit nozzles (psi): dP = 8.311e-5 * rho * q^2 / (Cd^2 * A^2)."""
    return round(8.311e-5 * mw_ppg * q_gpm ** 2 / (cd ** 2 * tfa_in2 ** 2), 1)


def pipe_loss(q_gpm, mw_ppg, pv_cp, yp, d_in, length_ft):
    """Bingham-plastic frictional loss inside pipe (psi), laminar or turbulent by the
    equivalent Reynolds number (turbulent above 2100)."""
    v = q_gpm / (2.448 * d_in ** 2)                                   # ft/s
    mu_a = pv_cp + 6.66 * yp * d_in / max(v, 1e-9)                    # apparent viscosity, cP
    re_ = 928 * mw_ppg * v * d_in / mu_a
    if re_ < 2100:
        g = pv_cp * v / (1500 * d_in ** 2) + yp / (225 * d_in)
    else:
        g = mw_ppg ** 0.75 * v ** 1.75 * pv_cp ** 0.25 / (1800 * d_in ** 1.25)
    return dict(dp_psi=round(g * length_ft, 1), velocity_ft_s=round(v, 2), reynolds=round(re_), regime="laminar" if re_ < 2100 else "turbulent")


def annulus_loss(q_gpm, mw_ppg, pv_cp, yp, d_hole_in, d_pipe_in, length_ft):
    """Bingham-plastic frictional loss in a concentric annulus (psi)."""
    dh = d_hole_in - d_pipe_in
    v = q_gpm / (2.448 * (d_hole_in ** 2 - d_pipe_in ** 2))
    mu_a = pv_cp + 5 * yp * dh / max(v, 1e-9)
    re_ = 757 * mw_ppg * v * dh / mu_a
    if re_ < 2100:
        g = pv_cp * v / (1000 * dh ** 2) + yp / (200 * dh)
    else:
        g = mw_ppg ** 0.75 * v ** 1.75 * pv_cp ** 0.25 / (1396 * dh ** 1.25)
    return dict(dp_psi=round(g * length_ft, 1), velocity_ft_min=round(60 * v, 1), reynolds=round(re_),
                regime="laminar" if re_ < 2100 else "turbulent")


def spp_estimate(q_gpm, mw_ppg, pv_cp, yp, tfa_in2, depth_ft, dp_od, dp_id, dp_len_ft, bha_od, bha_id, bha_len_ft,
                 hole_in, motor_dp_psi=0.0, surface_psi=0.0):
    """Standpipe pressure = surface equipment + drill pipe + BHA + motor/MWD + bit + annulus."""
    parts = {
        "drill_pipe": pipe_loss(q_gpm, mw_ppg, pv_cp, yp, dp_id, dp_len_ft)["dp_psi"],
        "bha_bore": pipe_loss(q_gpm, mw_ppg, pv_cp, yp, bha_id, bha_len_ft)["dp_psi"],
        "bit": bit_pressure_drop(mw_ppg, q_gpm, tfa_in2),
        "annulus_dp": annulus_loss(q_gpm, mw_ppg, pv_cp, yp, hole_in, dp_od, dp_len_ft)["dp_psi"],
        "annulus_bha": annulus_loss(q_gpm, mw_ppg, pv_cp, yp, hole_in, bha_od, bha_len_ft)["dp_psi"],
        "motor_and_tools": motor_dp_psi, "surface": surface_psi}
    ann = parts["annulus_dp"] + parts["annulus_bha"]
    return dict(spp_psi=round(sum(parts.values()), 0), breakdown=parts,
                ecd_ppg=round(mw_ppg + ann / (0.052 * max(depth_ft, 1.0)), 2))


def nozzles_to_explain(delta_spp_psi, mw_ppg, q_gpm, nozzles):
    """How many nozzles would have to plug to raise SPP by delta_spp (everything else
    unchanged)? Plugs the nozzles one at a time, largest first."""
    pairs = sorted([int(d) for n, d in re.findall(r"(\d+)\s*[xX]\s*(\d+)", nozzles) for _ in range(int(n))], reverse=True)
    tfa0 = tfa_from_nozzles(nozzles)
    base = bit_pressure_drop(mw_ppg, q_gpm, tfa0)
    out, tfa = [], tfa0
    for k, d in enumerate(pairs[:-1], 1):
        tfa -= math.pi / 4 * (d / 32.0) ** 2
        rise = bit_pressure_drop(mw_ppg, q_gpm, tfa) - base
        out.append(dict(plugged=k, tfa_in2=round(tfa, 3), spp_rise_psi=round(rise)))
    best = min(out, key=lambda r: abs(r["spp_rise_psi"] - delta_spp_psi)) if out else None
    return dict(base_bit_dp_psi=base, tfa_in2=tfa0, scenarios=out, closest=best)
