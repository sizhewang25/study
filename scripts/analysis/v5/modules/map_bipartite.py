"""Two maps of the bipartite graph: topology and flows.

Ported from v3's `map_bipartite.py`, onto v5's answer space and map
primitives. Renders what `build-bipartite-graph` produced. The two maps are
complements and neither replaces the other:

* **`bipartite_nodes_map.png` -- topology.** Where the VPs and TG sites are,
  against the TG grids and the cells they have to resolve. No edges, so nothing
  occludes the geometry. This is the figure that makes the occupied-grid counts
  visible: "134 VPs in 31 grids" is the honest denominator for any claim
  resting on independent observations, and a scalar does not show that two
  dozen of them sit in one metro.
* **`bipartite_flows_map.png` -- flows.** Every measured edge, so what the
  campaign actually covers reads directly. The topology map cannot show this
  and this map cannot show the topology, because the ink that carries the
  flows is the ink that hides the nodes.

v3 drew a third figure, `distance_cdf.png`. It is not ported; the CDFs it drew
are still written as `edge_length_cdf.csv` and `pairwise_distance_cdf.csv`.

The cells are `cells.cell_polygons`, the same unbounded Voronoi that
`plot-answer-space` draws, so the boundary here is the one `classify` scores
against.

Command: `plot-bipartite-graph`. Needs `build-answer-space` and
`build-bipartite-graph`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import mapping as M
from scripts.analysis.v5.modules.answer_space import AnswerSpace, site_n_scored
from scripts.analysis.v5.modules.bipartite import BipartiteGraph, load_bipartite, load_rung
from scripts.analysis.v5.modules.paths import RunPaths

NODES_PNG = "bipartite_nodes_map.png"
FLOWS_PNG = "bipartite_flows_map.png"

_FIGSIZE = (14.0, 8.0)

#: The flows are a density field rather than a category, so they carry no hue.
#: `INK_2` is also the cell ink, which is why `--no-flow-cells` exists.
_FLOW_COLOR = M.INK_2

#: Flow-line alpha and width. A near-complete bipartite graph puts thousands of
#: lines on one frame, so both sit at the low end: density has to come from
#: accumulation, not from each line being visible.
_FLOW_ALPHA = 0.18
_FLOW_LINEWIDTH = 0.25


def _new_axes(extent):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    M.draw_basemap(ax)
    return fig, ax


def _caption(ax, text: str) -> None:
    ax.annotate(
        text, xy=(0.5, -0.03), xycoords="axes fraction", ha="center", va="top",
        fontsize=8.5, color=M.INK_2,
    )


def _save(fig, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def extent_for(space: AnswerSpace, graph: BipartiteGraph) -> tuple[float, float, float, float]:
    """The frame, padded around **both** node sets.

    `plot-answer-space --auto-extent` frames on sites alone, which is right
    there; here a VP outside the site hull would be silently cropped, and an
    off-frame VP is exactly what the topology figure exists to show.
    """
    lats = np.concatenate([space.sites["site_lat"].to_numpy(), graph.vp_nodes["vp_lat"].to_numpy()])
    lons = np.concatenate([space.sites["site_lon"].to_numpy(), graph.vp_nodes["vp_lon"].to_numpy()])
    return M.auto_extent(lats, lons)


def _draw_sites_and_seeds(ax, space: AnswerSpace) -> None:
    import cartopy.crs as ccrs

    # Hollow where the run scores none of the site's TGs, as on the answer-space
    # map. Only a traffic-weighted run has any: its space is the mesh, and those
    # sites lost every flow, so they also carry no edge here.
    n_scored = site_n_scored(space).to_numpy()
    scored = n_scored > 0
    sites = space.sites
    ax.scatter(
        sites["site_lon"][scored], sites["site_lat"][scored], s=M.MARKER_AREA, c=M.INK,
        marker="o", linewidths=0, transform=ccrs.PlateCarree(), zorder=6,
        label=f"Site ({int(scored.sum()):,}; {int(n_scored.sum()):,} TGs)",
    )
    if (~scored).any():
        ax.scatter(
            sites["site_lon"][~scored], sites["site_lat"][~scored], s=M.MARKER_AREA,
            facecolors="none", edgecolors=M.INK, marker="o", linewidths=0.7,
            transform=ccrs.PlateCarree(), zorder=6,
            label=f"Mesh site, no scored TG ({int((~scored).sum())})",
        )
    ax.scatter(
        space.seeds["seed_lon"], space.seeds["seed_lat"], s=M.MARKER_AREA * 2.2, c=M.INK,
        marker="x", linewidths=0.9, transform=ccrs.PlateCarree(), zorder=7,
        label=f"Seed ({space.n_seeds})",
    )


def _draw_vps(ax, graph: BipartiteGraph, *, size: float) -> None:
    import cartopy.crs as ccrs

    # Last and largest: the VPs are what these maps add over the answer-space
    # map, and a white ring keeps a cluster of co-located VPs countable.
    ax.scatter(
        graph.vp_nodes["vp_lon"], graph.vp_nodes["vp_lat"], s=size, c=M.VP_COLOR,
        marker="^", alpha=0.95, edgecolors="white", linewidths=0.4,
        transform=ccrs.PlateCarree(), zorder=8, label=f"VP ({graph.n_vps:,})",
    )


def _grid_label(space: AnswerSpace) -> str:
    return f"healpix nside={space.nside} (~{space.grid_km:.0f} km grids)"


def plot_bipartite_nodes(
    space: AnswerSpace,
    graph: BipartiteGraph,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    vp_size: float = 22.0,
    cells: bool = True,
    vp_grids: bool = True,
) -> Path:
    """Nodes, grids and cells -- no edges."""
    extent = extent or extent_for(space, graph)
    fig, ax = _new_axes(extent)

    M.draw_grids(
        ax, space.grids["grid_id"].to_numpy(), space.nside, facecolor=M.TARGET_FILL,
        edgecolor=M.TARGET_EDGE, linewidth=0.5, alpha=0.80, zorder=3,
    )
    # VP grids, outlined only and above the fills so they stay readable where a
    # VP grid and a TG grid coincide.
    n_vp_grids = int(graph.vp_nodes["vp_grid_id"].nunique())
    if vp_grids:
        M.draw_grids(
            ax, graph.vp_nodes["vp_grid_id"].drop_duplicates().to_numpy(), space.nside,
            edgecolor=M.VP_COLOR, linewidth=0.7, zorder=3.5,
        )
    if cells:
        M.draw_cells(ax, CL.cell_polygons(space.seeds, extent), zorder=4)

    _draw_sites_and_seeds(ax, space)
    _draw_vps(ax, graph, size=vp_size)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    m = graph.meta
    vp_occ = m["nodes"]["vps"]["dispersion"]["occupancy_ratio"]
    e = m["edges"]
    ax.set_title(title or "Bipartite graph — topology", fontsize=12)
    _caption(
        ax,
        f"{_grid_label(space)} · {graph.n_vps:,} VPs in {n_vp_grids} grids "
        f"(occupancy {vp_occ:.2f}) · {graph.n_tgs:,} TGs at {len(space.sites)} sites "
        f"in {len(space.grids)} grids, {space.n_seeds} seeds\n"
        f"{e['n_edges']:,} measured edges · density {e['edge_density']:.3f} · "
        f"{e['connected_components']['n_components']} component(s) · "
        f"median nearest measured VP {e['nearest_vp_km']['observed']['percentiles']['p50']:.0f} km · "
        f"median max angular gap {m['angular']['max_angular_gap_deg']['percentiles']['p50']:.0f}°\n"
        "Blue outlines are VP-occupied grids — the denominator behind any "
        "independent-observation claim, since VPs sharing a grid produce "
        "near-identical constraints.",
    )
    return _save(fig, out_path)


def plot_bipartite_flows(
    space: AnswerSpace,
    graph: BipartiteGraph,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    max_segments: int | None = None,
    cells: bool = True,
    seed: int = 20260903,
) -> Path:
    """Every measured edge as a great-circle line.

    Lines are the **distinct** `(VP coord, TG coord)` pairs from
    `edge_segments.csv`, not one per edge, with width carrying multiplicity --
    see `bipartite.edge_segments` for why.
    """
    import cartopy.crs as ccrs
    from matplotlib.collections import LineCollection

    extent = extent or extent_for(space, graph)
    fig, ax = _new_axes(extent)

    seg = graph.edge_segments
    n_all = len(seg)
    sampled = max_segments is not None and n_all > int(max_segments)
    if sampled:
        seg = seg.sample(n=int(max_segments), random_state=seed).sort_index()

    if len(seg):
        paths = M.great_circle_segments(
            seg["vp_lat"].to_numpy(dtype=float), seg["vp_lon"].to_numpy(dtype=float),
            seg["tg_lat"].to_numpy(dtype=float), seg["tg_lon"].to_numpy(dtype=float),
        )
        mult = seg["n_edges"].to_numpy(dtype=float)
        ax.add_collection(
            LineCollection(
                list(paths), colors=_FLOW_COLOR,
                linewidths=_FLOW_LINEWIDTH * (1.0 + np.log10(np.maximum(mult, 1.0))),
                alpha=_FLOW_ALPHA, transform=ccrs.PlateCarree(), zorder=2.5,
            )
        )
    if cells:
        M.draw_cells(ax, CL.cell_polygons(space.seeds, extent), linewidth=0.6, zorder=4)

    _draw_sites_and_seeds(ax, space)
    _draw_vps(ax, graph, size=30.0)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    e = graph.meta["edges"]
    length = e["length_km"]["observed"]["percentiles"]
    sample_note = f" Sampled to {len(seg):,} of them (seed {seed})." if sampled else ""
    ax.set_title(title or "Bipartite graph — measured flows", fontsize=12)
    _caption(
        ax,
        f"{e['n_edges']:,} measured (VP, TG) edges, drawn as {n_all:,} geometrically "
        f"distinct great-circle lines; width carries how many edges each stands for."
        f"{sample_note}\n"
        f"Edge length p50 {length['p50']:,.0f} km, p90 {length['p90']:,.0f} km · "
        f"density {e['edge_density']:.3f} · "
        f"{e['connected_components']['n_components']} component(s)\n"
        "TG replicas at one site collapse into one line on purpose: one line per "
        "row would report them as heavier traffic.",
    )
    return _save(fig, out_path)


def build_for_run(
    run: RunPaths,
    *,
    nside: int = G.DEFAULT_NSIDE,
    extent: tuple[float, float, float, float] | None = None,
    analysis_root: Path | None = None,
    vp_size: float = 22.0,
    cells: bool = True,
    flow_cells: bool = True,
    vp_grids: bool = True,
    max_segments: int | None = None,
) -> tuple[Path, Path, BipartiteGraph]:
    """Both maps into the run's `bipartite-graph/healpix-<nside>/`.

    `extent=None` frames both node sets. `flow_cells` only matters when `cells`
    is on: either switch off removes the cells from the flow map, and only
    `cells` reaches the topology map, where the boundary is the point.
    """
    out_dir = run.bipartite_dir(nside, root=analysis_root)
    graph = load_bipartite(out_dir)
    space = load_rung(run, nside, analysis_root=analysis_root)
    label = f"nside {space.nside}"
    nodes = plot_bipartite_nodes(
        space, graph, out_dir / NODES_PNG, extent=extent,
        title=f"{run.run_id} — bipartite graph, topology ({label})",
        vp_size=vp_size, cells=cells, vp_grids=vp_grids,
    )
    flows = plot_bipartite_flows(
        space, graph, out_dir / FLOWS_PNG, extent=extent,
        title=f"{run.run_id} — bipartite graph, measured flows ({label})",
        max_segments=max_segments, cells=cells and flow_cells,
    )
    return nodes, flows, graph
