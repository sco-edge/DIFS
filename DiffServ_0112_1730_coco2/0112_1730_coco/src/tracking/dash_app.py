"""Dash-based real-time performance tracker for DIFS_OC_v7.

Provides an animated, interactive dashboard with:
  * Live line charts (Plotly) for each tracked metric over time
  * Legends showing metric names + units
  * Peak annotations with 2-decimal values + units
  * Baseline comparison reference lines
  * Auto-refresh every second via d.Interval
  * Manual reset support
"""

import json
from pathlib import Path

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from tracking.realtime_tracker import RealTimeTracker

UNITS = {
    "slo_violation_rate": "%",
    "cold_start_latency": "s",
    "hot_start_latency": "s",
    "error_to_normal_ratio": "",
}


def _format_val(v: float, key: str) -> str:
    """Format a metric value: 2 decimals + unit, percentage for slo_violation_rate."""
    if key == "slo_violation_rate":
        v = v * 100.0
    unit = UNITS.get(key, "")
    return f"{v:.2f}{unit}"


def _format_pct(pct, key: str) -> str:
    """Format improvement percentage with 2 decimals + optional + sign."""
    if key == "slo_violation_rate" and pct is not None:
        pct = pct * 100.0
    if pct is None:
        return "n/a"
    prefix = "+" if pct >= 0 else ""
    return f"{prefix}{pct:.2f}%"


def _compute_bar_width(pct, target_pct) -> float:
    """Compute progress-bar width percentage for a metric."""
    if target_pct is None or target_pct == 0:
        return 0.0
    if pct is None:
        return 0.0
    try:
        return abs((float(pct) / float(target_pct)) * 100.0)
    except Exception:
        return 0.0


def _build_results_table(cur: dict, base: dict, comp: dict, obj: dict):
    """Build a real-time results table showing current values vs baseline/targets."""
    metrics_keys = ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]
    metric_names = {
        "slo_violation_rate": "SLO Violation Rate",
        "cold_start_latency": "Cold Start Latency",
        "hot_start_latency": "Hot Start Latency",
        "error_to_normal_ratio": "Error / Normal Latency Ratio",
    }

    header = ["Metric", "Unit", "Current", "Baseline", "Improvement", "Target", "Status", "Last Updated"]
    rows = []
    for key in metrics_keys:
        info = obj.get(key, {}) if isinstance(obj, dict) else {}
        met = info.get("met") if isinstance(info, dict) else None
        target_pct = info.get("target_pct") if isinstance(info, dict) else None
        pct = comp.get(key) if isinstance(comp, dict) else None

        cur_val = cur.get(key) if isinstance(cur, dict) else None
        base_val = base.get(key) if isinstance(base, dict) else None

        if cur_val is None or not isinstance(cur_val, (int, float)):
            cur_str = "n/a"
        else:
            cur_str = _format_val(cur_val, key)

        if base_val is None or not isinstance(base_val, (int, float)):
            base_str = "n/a"
        else:
            base_str = _format_val(base_val, key)

        if pct is None or not isinstance(pct, (int, float)):
            imp_str = "n/a"
        else:
            imp_str = _format_pct(pct, key)

        if target_pct is None or not isinstance(target_pct, (int, float)):
            target_str = "n/a"
        else:
            target_str = f"{target_pct:.2f}%"

        if met is True:
            status_str = "✓ Met"
        elif met is False:
            status_str = "✗ Not Met"
        else:
            status_str = "n/a"

        rows.append([metric_names.get(key, key), UNITS.get(key, ""), cur_str, base_str, imp_str, target_str, status_str, cur.get("timestamp", "n/a") if isinstance(cur, dict) else "n/a"])

    table_header = [
        dash.html.Thead(dash.html.Tr([dash.html.Th(h, style={"backgroundColor": "#1a2230", "color": "#d7dde5", "border": "1px solid #2c3a4f", "padding": "8px 12px"}) for h in header]))
    ]
    table_body = []
    for r in rows:
        cells = []
        for i, val in enumerate(r):
            style = {"backgroundColor": "#18202c", "color": "#e5e7eb", "border": "1px solid #2c3a4f", "padding": "8px 12px", "fontSize": "13px"}
            if r[6] == "✓ Met":
                style["color"] = "#4ade80"
            elif r[6] == "✗ Not Met":
                style["color"] = "#f87171"
            if i in (2,3,4):
                style["textAlign"] = "right"
            cells.append(dash.html.Td(val, style=style))
        table_body.append(dash.html.Tr(cells))
    return dash.html.Table(table_header + [dash.html.Tbody(table_body)], style={"width": "100%", "borderCollapse": "collapse"})


def _app(project_root: str = ".") -> dash.Dash:
    root = Path(project_root).resolve()
    tracker = RealTimeTracker(root)

    app = dash.Dash(__name__, title="DIFS_OC_v7 Tracker")

    app.layout = dash.html.Div([
        dash.html.Div([
            dash.html.H1("DIFS_OC_v7 &mdash; Real-Time Performance Tracker"),
            dash.html.Div(id="dash-elapsed", style={"color": "#8b9bb0", "marginRight": "20px"}),
            dash.html.Button("Reset", id="dash-reset-n", n_clicks=0,
                             style={"marginLeft": "10px", "background": "#38bdf8", "color": "white",
                                    "border": "none", "borderRadius": "4px", "padding": "6px 12px"}),
        ], style={"padding": "16px 22px", "background": "#1a2230", "borderBottom": "1px solid #2c3a4f"}),

        dash.html.Div([
            dash.html.Div(id="dash-cards", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "12px", "padding": "18px 22px"}),
            dash.html.Div([
                dash.html.H3("Metric Charts"),
                dash.html.Div(id="dash-graphs", style={"height": "550px"}),
            ], style={"padding": "18px 22px", "borderLeft": "1px solid #2c3a4f"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"}),

        dash.html.Div([
            dash.html.H3("Current Results", style={"margin": "0 0 10px", "color": "#d7dde5"}),
            dash.html.Div(id="dash-table", style={"overflowX": "auto"}),
        ], style={"padding": "18px 22px", "borderTop": "1px solid #2c3a4f", "background": "#151b24"}),

        dash.html.Div(id="dash-foot", style={"padding": "12px 22px", "color": "#8b9bb0", "fontSize": "11px"}),
        dash.dcc.Interval(id="dash-interval", interval=1000, n_intervals=0, disabled=False),
    ])
    # API route for real-time metrics
    @app.server.route("/api/metrics")
    def metrics_api():
        try:
            snapshot = tracker.snapshot()
            return json.dumps(snapshot), 200, {"Content-Type": "application/json"}
        except Exception as e:
            return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _snapshot_data() -> dict:
        try:
            return tracker.snapshot()
        except Exception:
            return {"history": [], "current": {}, "baseline": {}, "objectives": {}, "metrics_file": str(tracker.metrics_file)}

    def _build_cards(cur: dict, base: dict, comp: dict, obj: dict) -> list:
        """Build card divs for each of the 4 metrics."""
        cards = []
        metrics_keys = ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]
        for key in metrics_keys:
            info = obj.get(key, {})
            met = info.get("met", False)
            pct = comp.get(key)

            # formatted value + unit
            val_str = _format_val(cur.get(key, 0), key)

            # improvement percentage string
            pct_str = _format_pct(pct, key)

            # progress-bar width: % of target achieved
            target_pct = info.get("target_pct")
            bar_width = _compute_bar_width(pct, target_pct)

            cards.append(dash.html.Div([
                dash.html.H2(f"{key} ({UNITS.get(key, '')})", style={"margin": "0 0 6px", "fontSize": "12px", "color": "#8b9bb0"}),
                dash.html.Div(val_str, style={"fontSize": "22px", "fontWeight": "600"}),
                dash.html.Div(pct_str, style={"fontSize": "11px", "marginTop": "4px",
                              "color": "#4ade80" if met else "#f87171"}),
                dash.html.Div(style={"height": "6px", "background": "#0b0f15", "borderRadius": "3px", "marginTop": "8px", "overflow": "hidden"},
                              children=dash.html.Div(style={"width": f"{bar_width:.1f}%", "height": "100%", "background": "#38bdf8"})),
            ], style={"background": "#18202c", "border": "1px solid #2c3a4f", "borderRadius": "8px", "padding": "12px"}))
        return cards

    def _build_graphs(history: list) -> list:
        """Build a Plotly figure for each metric from the history list."""
        metrics_keys = ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]

        if not history or len(history) < 2:
            figs = []
            for _key in metrics_keys:
                fig = go.Figure()
                fig.add_annotation(text="waiting for samples...", x=0.5, y=0.5, showarrow=False,
                                   font={"size": 14})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#8b9bb0", margin=dict(l=20, r=20, t=40, b=20))
                figs.append(fig)
            return figs

        figures = []
        for key in metrics_keys:
            vals = [h.get("metrics", {}).get(key) for h in history if isinstance(h.get("metrics", {}).get(key), (int, float))]
            if not vals:
                fig = go.Figure()
                fig.add_annotation(text="no data", x=0.5, y=0.5, showarrow=False,
                                   font={"size": 14})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#8b9bb0", margin=dict(l=20, r=20, t=40, b=20))
                figures.append(fig)
                continue

            # Normalize each metric independently (own min/max span)
            norm_vals = [v * 100.0 if key == "slo_violation_rate" else v for v in vals]
            min_v, max_v = min(norm_vals), max(norm_vals)
            span = (max_v - min_v) or 1e-9
            times = list(range(len(norm_vals)))

            # Find global peak for annotation
            peak_idx = max(range(len(norm_vals)), key=lambda i: norm_vals[i])
            peak_val = norm_vals[peak_idx]

            # Baseline reference if available
            base_val = None
            for h in history[-5:]:
                bh = h.get("metrics", {}).get(key)
                if bh is not None:
                    base_val = bh * 100.0 if key == "slo_violation_rate" else bh
                    break

            # Create figure
            fig = go.Figure()

            # Main line trace
            fig.add_trace(go.Scatter(
                x=times, y=norm_vals,
                mode='lines+markers',
                line=dict(color="#38bdf8", width=2),
                marker=dict(size=4, color="#fbbf24"),
                name=f"{key} ({UNITS[key]})",
                hovertemplate=f'<b>Sample %{{x}}</b><br>{key}: %{{y:.2f}}{UNITS[key]}<extra></extra>'
            ))

            # Baseline reference dashed line + annotation
            if base_val is not None:
                fig.add_hrect(y0=base_val, y1=base_val, fillcolor="rgba(251,191,36,0.3)", layer="below", line_width=0)
                fig.add_annotation(x=0, y=base_val, text=f"baseline {base_val:.2f}{UNITS[key]}",
                                   showarrow=False, font=dict(size=9, color="#fbbf24"),
                                   xref="x", yref="y")

            # Peak annotation
            peak_display = _format_val(peak_val / 100.0 if key == "slo_violation_rate" else peak_val, key)
            fig.add_annotation(x=peak_idx, y=peak_val,
                               text=f"peak {peak_display}",
                               showarrow=True, arrowhead=2,
                               font=dict(size=9, color="#e5e7eb"),
                               bgcolor="rgba(16,23,34,0.8)", bordercolor="#38bdf8", borderwidth=1)

            if key == "slo_violation_rate":
                y_title = "SLO Violation Rate (%)"
            elif key == "cold_start_latency":
                y_title = "Cold Start Latency (s)"
            elif key == "hot_start_latency":
                y_title = "Hot Start Latency (s)"
            elif key == "error_to_normal_ratio":
                y_title = "Error / Normal Latency Ratio"
            else:
                y_title = f"{key} ({UNITS[key]})"
            x_title = "Time / Samples"
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8b9bb0",
                title=dict(text=f"{key.replace('_', ' ').title()}", x=0.5, font=dict(size=12, color="#d7dde5")),
                margin=dict(l=65, r=40, t=45, b=55),
                xaxis=dict(showgrid=True, gridcolor="#2c3a4f", zeroline=False, showticklabels=True, title=x_title,
                           tickfont=dict(color="#8b9bb0"), titlefont=dict(color="#d7dde5")),
                yaxis=dict(showgrid=True, gridcolor="#2c3a4f", title=y_title,
                           tickfont=dict(color="#8b9bb0"), titlefont=dict(color="#d7dde5")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(color="#8b9bb0"), bgcolor="rgba(0,0,0,0)"),
                showlegend=True,
            )

            figures.append(fig)

        return figures

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    @app.callback(
        Output("dash-elapsed", "children"),
        Output("dash-cards", "children"),
        Output("dash-graphs", "children"),
        Output("dash-table", "children"),
        Output("dash-foot", "children"),
        Input("dash-interval", "n_intervals"),
        Input("dash-reset-n", "n_clicks"),
        prevent_initial_call=False,
    )
    def update(n_intervals, reset_n):
        # Reset on button click
        if reset_n and reset_n > 0:
            tracker.reset()
            snap = tracker.snapshot()
        else:
            snap = _snapshot_data()

        cur = snap.get("current", {})
        base = snap.get("baseline", {})
        comp = snap.get("comparison", {})
        obj = snap.get("objectives", {})
        history = snap.get("history", [])

        elapsed = snap.get("current", {}).get("elapsed_time", 0) or 0

        cards = _build_cards(cur, base, comp, obj)
        figures = _build_graphs(history)
        table = _build_results_table(cur, base, comp, obj)

        graph_children = dash.html.Div([
            dash.dcc.Graph(figure=fig, style={"height": "260px", "width": "100%"})
            for fig in figures
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "overflowY": "auto", "height": "550px"})

        foot = f"state file: {snap.get('metrics_file', '')}  |  elapsed {elapsed:.1f}s"

        return dash.html.Div(elapsed), cards, graph_children, table, foot

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    app.tracker = tracker

    return app


if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description="DIFS_OC_v7 Dash tracker")
    parser.add_argument("--project-root", default=".", help="v7 project root")
    parser.add_argument("--port", type=int, default=8051, help="HTTP port")
    args = parser.parse_args()
    project_root = args.project_root
    port = args.port
    app = _app(project_root)
    try:
        app.run_server(host="127.0.0.1", port=port, debug=False)
    except Exception:
        app.server.run(host="127.0.0.1", port=port, debug=False)