"""Unit handling. Field data mix units silently (hook loads typed as 93 or 93000,
torque as 4.5 or 4500, mud weight as ppg or s.g.), so values are checked against the
range a quantity can physically take before they are converted."""

CONVERT = {
    ("lb", "klb"): 1e-3, ("klb", "lb"): 1e3,
    ("ft-lb", "kft-lb"): 1e-3, ("kft-lb", "ft-lb"): 1e3,
    ("ft", "m"): 0.3048, ("m", "ft"): 1 / 0.3048,
    ("ppg", "sg"): 1 / 8.345, ("sg", "ppg"): 8.345,
    ("ppg", "psi/ft"): 0.052, ("psi/ft", "ppg"): 1 / 0.052,
    ("ppg", "lb/ft3"): 7.4805, ("lb/ft3", "ppg"): 1 / 7.4805,
    ("bar", "psi"): 14.5038, ("psi", "bar"): 1 / 14.5038,
    ("in", "mm"): 25.4, ("mm", "in"): 1 / 25.4,
}

# Physically plausible ranges in the canonical unit, used to read undeclared units.
RANGES = {
    "hook_load_klb": (20.0, 1500.0),      # empty block ~40 klb .. heavy casing strings
    "torque_kftlb": (0.0, 60.0),
    "mud_weight_ppg": (6.5, 20.0),
}


def convert(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    k = CONVERT.get((from_unit, to_unit))
    if k is None:
        raise ValueError(f"no conversion {from_unit} -> {to_unit}; known: {sorted(set(u for p in CONVERT for u in p))}")
    return value * k


def read_hook_load(value, unit=""):
    """Hook load in klb from a value whose unit may be missing. A crew writes '93'
    (klb) on one tour and '93000' (lb) on the next: only one reading is physical."""
    if unit == "klb":
        return value, "declared klb"
    if unit == "lb":
        return value / 1000.0, "declared lb"
    lo, hi = RANGES["hook_load_klb"]
    as_klb, as_lb = lo <= value <= hi, lo <= value / 1000.0 <= hi
    if as_klb and not as_lb:
        return value, "undeclared: only klb is physical"
    if as_lb and not as_klb:
        return value / 1000.0, "undeclared: only lb is physical"
    return None, "undeclared and ambiguous"


def read_torque(value, unit=""):
    """Torque in kft-lb; ft-lb values are > 100 for any rotating string."""
    if unit in ("kft-lb", "kft_lb"):
        return value, "declared kft-lb"
    if unit in ("ft-lb", "ft_lb"):
        return value / 1000.0, "declared ft-lb"
    if value > 100:
        return value / 1000.0, "undeclared: > 100 so ft-lb"
    return value, "undeclared: <= 100 so kft-lb"


def read_mud_weight(value, unit="ppg"):
    """Mud weight in ppg. A value of 1.0-2.4 labelled ppg is almost certainly s.g."""
    lo, hi = RANGES["mud_weight_ppg"]
    if unit == "sg" or (1.0 <= value <= 2.4):
        return round(value * 8.345, 2), "read as specific gravity"
    if lo <= value <= hi:
        return value, "ppg, in range"
    return None, f"{value} is not a physical mud weight in ppg or s.g."
