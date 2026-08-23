"""Shared mesh-map rendering helpers for the figures/ scripts.

Rendering conventions are matched to MacMillan & Ouellette (arXiv:2512.24440):
equirectangular (PlateCarree) with Natural Earth 50m coastlines. Their Fig 3 draws
grid-locked features as DISCRETE MESH NODES (smoothing erases the node-level contrast
that defines grid-lock); their Figs 2/4 draw physical features as smooth fields.
Use scatter for the former, `to_grid` + pcolormesh for the latter.

Needs cartopy -> run in the JAX env.
"""
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.spatial import cKDTree

PC = ccrs.PlateCarree()

latlon = np.load("/tmp/m5_latlon.npy")
LAT, LON = latlon[:, 0], latlon[:, 1]
DEG = np.load("/tmp/m5_degree.npy")

# ---- mesh -> lat/lon interpolation (inverse distance, 4 great-circle neighbours) ----
GLON = np.arange(-180.0, 180.0, 0.5) + 0.25
GLAT = np.arange(-89.75, 90.0, 0.5)


def _unit(la, lo):
    a, o = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)


_MG2, _MG1 = np.meshgrid(GLON, GLAT)
_D, _I = cKDTree(_unit(LAT, LON)).query(_unit(_MG1.ravel(), _MG2.ravel()), k=4)


def to_grid(vals, power=2.0):
    """Interpolate a per-mesh-node field onto the 0.5 deg lat-lon grid."""
    w = 1.0 / np.maximum(_D, 1e-9) ** power
    return ((vals[_I] * w).sum(1) / w.sum(1)).reshape(_MG1.shape)


def worldmap(ax, title, coast="#aaa", coast_lw=0.5, coast_alpha=1.0, coast_z=1,
             ticks=True, title_size=10):
    """Equirectangular world axes.

    NOTE: cartopy's gridlines(draw_labels=True) pushes the axes title to y=inf, which
    bbox_inches='tight' then crops away -- so use real PlateCarree ticks for the labels
    and keep the gridlines label-free.
    """
    ax.set_global()
    ax.coastlines(resolution="50m", linewidth=coast_lw, color=coast, alpha=coast_alpha,
                  zorder=coast_z)
    ax.gridlines(draw_labels=False, linewidth=0.4, color="#d9d9d9", zorder=0,
                 xlocs=[-180, -90, 0, 90, 180], ylocs=[-60, -30, 0, 30, 60])
    if ticks:
        ax.set_xticks([-180, -90, 0, 90, 180], crs=PC)
        ax.set_yticks([-60, -30, 0, 30, 60], crs=PC)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=title_size, weight="bold", loc="left")


def participation_ratio(A):
    """Effective number of mesh nodes lit, per feature. 10242 = perfectly uniform."""
    A = np.clip(A, 0, None)
    s1, s2 = A.sum(-1), (A ** 2).sum(-1)
    return np.where(s2 > 0, s1 ** 2 / np.maximum(s2, 1e-30), np.nan)
