"""Derived slugs: comparisons across groups, evaluated at render time."""

from __future__ import annotations

from scripts.analysis.v5.paper_values.registry import derive

derive("tf.err.ratio.p50.mesh-over-tf.s-p", ["err.p50.s-p", "tf.err.p50.s-p"], "x1", lambda mesh, tf: mesh / tf,
       section="App. E", note="S-P's median error falls by this factor from the mesh to the TF datasets")
