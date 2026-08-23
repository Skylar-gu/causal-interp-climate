"""Config for the 2021 Pacific-NW heat-dome / blocking necessity experiment.

Frozen before results (see docs/notes/spec_heatdome_blocking.md). The event: the record
omega-block over western North America, peak ~26-29 Jun 2021 (Lytton BC 49.6 C). A
persistent high-latitude z500 ridge with extreme heat underneath.

Longitudes stored in -180..180 here; the ERA5 grid is 0..360 so selection code must
normalize (past bug).

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: none beyond the arguments above
Outputs: printed report
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.heatdome.heatdome_config
"""
import numpy as np

# --- event ---
IC = "2021-06-24"                       # ~24-25 Jun IC (spec)
H = 24                                  # +144h == 6 days (roll through the ridge peak Jun 27-29 and decay)
BOX = dict(lat=(45, 62), lon=(-135, -100))     # W North America ridge/heat box
PEAK_WINDOW_H = (72, 120)               # +72h..+120h == Jun 27 00Z .. Jun 29 00Z (ridge peak)

# atlas blocking/ridge candidates homed over W-NA (spec-frozen)
#   1789 (+53,-115) teleconnection/mode ; 492 (+53,-120) ; 2930 (+56,-118) 
#   1703 (+60,-134) ; 1036 (+50,-104)   climatology/clock
CANDS = [1789, 492, 2930, 1703, 1036]

RADIUS_KM = 1500.0                      # disk around the ridge centre for the local counterfactual

# quiet, no-heatdome late-June analog years -> the NORMAL blocking-feature reference.
# Code skips any analog where the heat-dome feature actually fires strongly (ridge present).
ANALOGS = ["2012-06-27", "2013-06-27", "2014-06-27", "2016-06-27", "2019-06-27", "2011-06-27"]

# a NON-block IC (spec control: same feature ablated when there is no ridge should do little).
# Mid-May 2021, W-NA in a normal westerly regime (no omega block).
NONBLOCK_IC = "2021-05-15"

G = 9.80665                             # geopotential -> geopotential height (m)

def norm_lon(lon):
    """-180..180 -> 0..360 for ERA5 selection."""
    return np.where(np.asarray(lon) < 0, np.asarray(lon) + 360.0, np.asarray(lon))
