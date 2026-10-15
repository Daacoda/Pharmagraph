r"""
make_manuscript_figures.py
==========================
Builds manuscript-ready figures and tables from the CSV files already written to output/
by exp1 / exp2 / exp3 / exp4a / ablation / t1_task_graphsage.  READ-ONLY: it never trains,
never touches the graph, never modifies an existing file.  Missing inputs are skipped.

Run from the project root:
    py src\make_manuscript_figures.py
    py src\make_manuscript_figures.py --out-dir output --fig-dir output\manuscript

Needs: pandas, numpy, matplotlib   (pip install matplotlib)

Writes (output\manuscript\):
    figures\fig1_class_imbalance.png/.pdf     AUROC and AUPRC vs negative:positive ratio
    figures\fig2_popularity_audit.png/.pdf    internal audit + contraindication-edge leak test
    figures\fig3_temporal_ot.png/.pdf         Open Targets temporal validation (pairwise AUROC)
    figures\fig4_per_drug_ranking.png/.pdf    per-drug ranking with bootstrap CIs and paired differences
    figures\fig5_ranking_by_drug_degree.png/.pdf   cold vs warm drugs
    figures\fig6_ablation.png/.pdf            ablation (mean +- std over seeds)
    figures\fig7_go_tasks.png/.pdf            t1 GO tasks
    tables\table*.csv  and  table*.md         same numbers, ready to paste
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9a9994"
# a scorer keeps the same colour in every figure (colour follows the entity, not its rank)
FIXED = {"model": CAT[0], "graphsage": CAT[0], "random": GRAY,
         "disease_popularity": CAT[1], "term_popularity": CAT[1],
         "degree_sum": CAT[2], "degree_baseline": CAT[2],
         "fingerprint_knn": CAT[3], "knn": CAT[3], "ontology_neighbor": CAT[3],
         "shared_protein_aa": CAT[4], "adamic_adar": CAT[4], "shared_protein_adamic_adar": CAT[4],
         "drug_degree": CAT[6], "disease_degree": CAT[7]}
_dyn = {}


def color_of(name):
    key = str(name)
    if key in FIXED:
        return FIXED[key]
    if key not in _dyn:
        used = set(FIXED.values()) | set(_dyn.values())
        free = [c for c in CAT if c not in used]
        _dyn[key] = free[0] if free else GRAY
    return _dyn[key]


def pretty(name):
    return {"model": "GraphSAGE", "degree_sum": "Degree sum", "disease_popularity": "Disease popularity",
            "term_popularity": "Term popularity", "fingerprint_knn": "Fingerprint kNN",
            "ontology_neighbor": "Ontology neighbour", "random": "Random",
            "drug_degree": "Drug degree", "disease_degree": "Disease degree",
            "degree_baseline": "Degree baseline"}.get(str(name), str(name).replace("_", " "))


plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
})


def clean(ax, grid="y"):
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid)
    ax.tick_params(length=2.5, width=0.6)


def panel_letter(ax, letter):
    ax.text(-0.02, 1.2, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load(out_dir, fname):
    p = os.path.join(out_dir, fname)
    if not os.path.exists(p):
        print(f"  skip: {p} not found")
        return None
    return pd.read_csv(p)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def pick_config(configs):
    configs = [c for c in pd.unique(configs)]
    for c in configs:
        if "phase>=1" in str(c) and "ot2111" in str(c):
            return c
    return configs[0] if configs else None


def short_cfg(c):
    c = str(c).replace("new_vs_ot2111", "new vs OT 21.11").replace("new_vs_t0", "new vs t0")
    return c.replace(" | ", "\n")


def write_table(df, tab_dir, name, float_fmt="{:.3f}"):
    os.makedirs(tab_dir, exist_ok=True)
    df.to_csv(os.path.join(tab_dir, f"{name}.csv"), index=False)
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else float_fmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    with open(os.path.join(tab_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {name}.csv / .md")


def grouped_bars(ax, cats, series, values, errs=None, label_values=True, width=0.8, fmt="{:.2f}", fs=6):
    """values[series][cat] -> float. One bar per (cat, series), thin bars, 2 px gap."""
    n = len(series)
    w = width / n
    x0 = np.arange(len(cats))
    for i, s in enumerate(series):
        xs = x0 - width / 2 + w * (i + 0.5)
        vs = np.array([values[s].get(c, np.nan) for c in cats], dtype=float)
        es = None if errs is None else np.array([errs[s].get(c, np.nan) for c in cats], dtype=float)
        ax.bar(xs, vs, width=w * 0.82, color=color_of(s), zorder=3, linewidth=0)
        if es is not None and np.isfinite(es).any():
            ax.errorbar(xs, vs, yerr=np.nan_to_num(es), fmt="none", ecolor=INK2, elinewidth=0.7,
                        capsize=1.5, capthick=0.7, zorder=4)
        if label_values:
            for xv, v, e in zip(xs, vs, np.nan_to_num(es) if es is not None else np.zeros(len(vs))):
                if np.isfinite(v):
                    ax.text(xv, v + e + 0.008, fmt.format(v), ha="center", va="bottom", fontsize=fs, color=INK2,
                            rotation=90)
    ax.set_xticks(x0)
    ax.set_xticklabels(cats)
    return x0


def legend_series(ax_or_fig, series, loc="upper center", ncol=None, **kw):
    handles = [Patch(facecolor=color_of(s), label=pretty(s)) for s in series]
    ncol = ncol or len(handles)
    return ax_or_fig.legend(handles=handles, loc=loc, ncol=ncol, **kw)


# --------------------------------------------------------------------------- figures
def fig1(out_dir, fig_dir, tab_dir):
    d = load(out_dir, "exp1_class_imbalance_summary.csv")
    if d is None:
        return
    modes = list(pd.unique(d.neg_mode))
    labels = list(d.sort_values("ratio").ratio_label.unique())
    x = np.arange(len(labels))
    mcol = {m: CAT[i] for i, m in enumerate(modes)}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for m in modes:
        g = d[d.neg_mode == m].set_index("ratio_label").reindex(labels)
        axs[0].errorbar(x, g.auroc_mean, yerr=g.auroc_std.fillna(0), color=mcol[m], lw=1.8, marker="o", ms=4,
                        mec="white", mew=0.8, capsize=2, label=str(m).replace("_", " "))
        axs[1].plot(x, g.auprc_mean, color=mcol[m], lw=1.8, marker="o", ms=4, mec="white", mew=0.8,
                    label=str(m).replace("_", " "))
    base = d.groupby("ratio_label").random_auprc_baseline.mean().reindex(labels)
    axs[1].plot(x, base, color=GRAY, lw=1.2, ls=(0, (4, 3)), label="chance (prevalence)")
    axs[0].set_ylim(max(0.5, d.auroc_mean.min() - 0.05), 1.0)
    axs[1].set_ylim(0, 1.0)
    axs[0].set_title("AUROC vs. class imbalance")
    axs[1].set_title("AUPRC vs. class imbalance")
    for a, yl, L in zip(axs, ("AUROC", "AUPRC"), "AB"):
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_xlabel("Negatives : positives")
        a.set_ylabel(yl)
        clean(a)
        panel_letter(a, L)
    axs[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, fig_dir, "fig1_class_imbalance")
    cols = [c for c in ("neg_mode", "ratio_label", "n_pos", "n_neg", "prevalence", "auroc_mean", "auroc_std",
                        "auprc_mean", "random_auprc_baseline") if c in d.columns]
    write_table(d.sort_values(["neg_mode", "ratio"])[cols], tab_dir, "table1_class_imbalance", "{:.4f}")


def fig2(out_dir, fig_dir, tab_dir):
    a = load(out_dir, "exp3_internal_baseline.csv")
    lk = load(out_dir, "exp4a_leak_test.csv")
    if a is None and lk is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.7, 1]})
    if a is not None:
        subs = list(pd.unique(a["subset"]))
        scs = ["model"] + [s for s in pd.unique(a.scorer) if s != "model"] if "model" in set(a.scorer) \
            else list(pd.unique(a.scorer))
        vals = {s: dict(zip(a[a.scorer == s]["subset"], a[a.scorer == s].auroc)) for s in scs}
        grouped_bars(axs[0], [s.replace("_", "\n") for s in subs], scs,
                     {s: {k.replace("_", "\n"): v for k, v in vals[s].items()} for s in scs}, fs=5.5)
        axs[0].axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        axs[0].set_ylim(0.4, 1.12)
        axs[0].set_ylabel("AUROC (internal test split)")
        axs[0].set_title("Popularity audit")
        axs[0].legend(handles=[Patch(facecolor=color_of(s), label=pretty(s)) for s in scs], ncol=3,
                      loc="upper center", bbox_to_anchor=(0.5, -0.4), fontsize=7)
        clean(axs[0])
        panel_letter(axs[0], "A")
        piv = a.pivot(index="scorer", columns="subset", values="auroc").reset_index()
        piv.insert(0, "scorer_name", piv.pop("scorer").map(pretty))
        write_table(piv.rename(columns={"scorer_name": "scorer"}), tab_dir, "table2_popularity_audit", "{:.3f}")
    if lk is not None:
        subs = list(pd.unique(lk["subset"]))
        x = np.arange(len(subs))
        w = 0.36
        axs[1].bar(x - w / 2, lk.auroc_visible, w * 0.9, color=CAT[0], zorder=3, label="Edges visible")
        axs[1].bar(x + w / 2, lk.auroc_hidden, w * 0.9, color=CAT[1], zorder=3, label="Edges hidden")
        for xi, v in zip(x - w / 2, lk.auroc_visible):
            axs[1].text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90, color=INK2)
        for xi, v in zip(x + w / 2, lk.auroc_hidden):
            axs[1].text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90, color=INK2)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([s.replace("_", "\n") for s in subs])
        lo = float(min(lk.auroc_visible.min(), lk.auroc_hidden.min()))
        axs[1].set_ylim(max(0.5, lo - 0.08), 1.0)
        axs[1].set_ylabel("AUROC")
        axs[1].set_title("Contraindication-edge test")
        axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=1, fontsize=7)
        clean(axs[1])
        panel_letter(axs[1], "B")
        write_table(lk, tab_dir, "table3_leak_test", "{:.4f}")
    fig.tight_layout()
    save(fig, fig_dir, "fig2_popularity_audit")


def fig3(out_dir, fig_dir, tab_dir):
    d = load(out_dir, "exp2_temporal_summary.csv")
    if d is None:
        return
    if "auroc_mean" not in d.columns:
        d = d.rename(columns={"auroc": "auroc_mean"})
    lab = "1:1" if "1:1" in set(d.ratio_label) else d.sort_values("ratio").ratio_label.iloc[0]
    d = d[d.ratio_label == lab]
    modes = list(pd.unique(d.neg_mode))
    cfgs = list(pd.unique(d.config))
    scs = ["model"] + [s for s in pd.unique(d.scorer) if s != "model"] if "model" in set(d.scorer) \
        else list(pd.unique(d.scorer))
    fig, axs = plt.subplots(1, len(modes), figsize=(3.6 * len(modes), 3.0), squeeze=False, sharey=True)
    for ax, m, L in zip(axs[0], modes, "ABCD"):
        g = d[d.neg_mode == m]
        vals = {s: dict(zip(g[g.scorer == s].config, g[g.scorer == s].auroc_mean)) for s in scs}
        errs = {s: dict(zip(g[g.scorer == s].config, g[g.scorer == s].get("auroc_std", pd.Series(dtype=float))))
                for s in scs}
        grouped_bars(ax, [short_cfg(c) for c in cfgs], scs,
                     {s: {short_cfg(k): v for k, v in vals[s].items()} for s in scs},
                     {s: {short_cfg(k): v for k, v in errs[s].items()} for s in scs}, fs=6)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"{str(m).replace('_', ' ')} negatives ({lab})")
        clean(ax)
        panel_letter(ax, L)
        ax.tick_params(axis="x", labelsize=6.5)
    axs[0][0].set_ylabel("AUROC (new Open Targets positives)")
    fig.legend(handles=[Patch(facecolor=color_of(s), label=pretty(s)) for s in scs], loc="lower center",
               ncol=len(scs), fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, fig_dir, "fig3_temporal_ot")
    piv = d.pivot_table(index=["config", "neg_mode"], columns="scorer", values="auroc_mean").reset_index()
    write_table(piv, tab_dir, "table4_temporal_pairwise_auroc", "{:.3f}")


def fig4(out_dir, fig_dir, tab_dir):
    b = load(out_dir, "exp4a_bootstrap.csv")
    if b is None:
        return
    cfg = pick_config(b.config)
    b = b[b.config == cfg]
    bucket = "ALL" if "ALL" in set(b.drug_bucket) else b.drug_bucket.iloc[0]
    b = b[b.drug_bucket == bucket]
    pt = b[~b.metric.str.contains(r"\(model - scorer\)")]
    df = b[b.metric.str.contains(r"\(model - scorer\)")]
    order_metric = "within_drug_auc"
    sc_order = pt[pt.metric == order_metric].sort_values("mean", ascending=False).scorer.tolist()
    if "model" in sc_order:
        sc_order = ["model"] + [s for s in sc_order if s != "model"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    y = np.arange(len(sc_order))[::-1]
    for ax, met, ttl, xl, L in ((axs[0], "within_drug_auc", "Within-drug AUC", "AUC (0.5 = chance)", "A"),
                                (axs[1], "hits@100", "Hits@100", "Fraction of queries", "B")):
        g = pt[pt.metric == met].set_index("scorer").reindex(sc_order)
        for yi, s in zip(y, sc_order):
            r = g.loc[s]
            ax.plot([r.lo, r.hi], [yi, yi], color=color_of(s), lw=2, solid_capstyle="round", zorder=3)
            ax.plot(r["mean"], yi, "o", color=color_of(s), ms=6, mec="white", mew=0.9, zorder=4)
            ax.text(r.hi + (g[["hi"]].max().iloc[0] - g[["lo"]].min().iloc[0]) * 0.03, yi, f"{r['mean']:.3f}",
                    va="center", fontsize=6.5, color=INK2)
        if met == "within_drug_auc":
            ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(ttl)
        ax.set_xlabel(xl)
        clean(ax, "x")
        panel_letter(ax, L)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([pretty(s) for s in sc_order])
    ax = axs[2]
    met = "within_drug_auc (model - scorer)"
    g = df[df.metric == met].set_index("scorer")
    ys = [yi for yi, s in zip(y, sc_order) if s in g.index]
    for yi, s in zip(y, sc_order):
        if s not in g.index:
            continue
        r = g.loc[s]
        ax.plot([r.diff_lo, r.diff_hi], [yi, yi], color=color_of(s), lw=2, solid_capstyle="round", zorder=3)
        ax.plot(r.diff_model_minus, yi, "o", color=color_of(s), ms=6, mec="white", mew=0.9, zorder=4)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_title("Model minus baseline")
    ax.set_xlabel("Paired diff. in within-drug AUC")
    clean(ax, "x")
    panel_letter(ax, "C")
    fig.suptitle(f"Per-drug ranking of new indications  [{short_cfg(cfg).replace(chr(10), ', ')}]; "
                 "drug-level bootstrap 95% CI", fontsize=7.5, color=INK2, x=0.01, ha="left", y=0.98)
    fig.tight_layout()
    save(fig, fig_dir, "fig4_per_drug_ranking")
    rows = []
    for s in sc_order:
        row = {"scorer": pretty(s)}
        for m in ("within_drug_auc", "hits@100", "mrr"):
            r = pt[(pt.scorer == s) & (pt.metric == m)]
            if len(r):
                r = r.iloc[0]
                row[m] = f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]"
        rows.append(row)
    write_table(pd.DataFrame(rows), tab_dir, "table5_per_drug_ranking")


def fig5(out_dir, fig_dir, tab_dir):
    d = load(out_dir, "exp3_summary.csv")
    if d is None:
        return
    d = d[d.config == pick_config(d.config)]
    d = d[(d.stratum == "drug_bucket") & (d.bucket != "ALL")]
    if d.empty:
        return
    want = [s for s in ("model", "disease_popularity", "random") if s in set(d.scorer)] or list(pd.unique(d.scorer))
    buckets = sorted(pd.unique(d.bucket), key=natural_key)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, met, ttl, L in ((axs[0], "within_drug_auc", "Within-drug AUC", "A"), (axs[1], "hits@100", "Hits@100", "B")):
        vals = {s: dict(zip(d[d.scorer == s].bucket, d[d.scorer == s][met])) for s in want}
        grouped_bars(ax, [str(b) for b in buckets], want, {s: {str(k): v for k, v in vals[s].items()} for s in want},
                     fmt="{:.2f}" if met == "within_drug_auc" else "{:.3f}")
        if met == "within_drug_auc":
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.set_ylim(0.3, 1.05)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.set_title(ttl)
        ax.set_xlabel("Drug degree bucket (t0 indications)")
        clean(ax)
        panel_letter(ax, L)
    fig.legend(handles=[Patch(facecolor=color_of(s), label=pretty(s)) for s in want], loc="lower center",
               ncol=len(want), fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, fig_dir, "fig5_ranking_by_drug_degree")
    cols = [c for c in ("scorer", "bucket", "n_queries", "n_drugs", "hits@10", "hits@100", "mrr",
                        "within_drug_auc") if c in d.columns]
    t = d[cols].copy()
    t["_k"] = t["bucket"].map(lambda b: tuple(str(x) for x in natural_key(b)))
    t = t.sort_values(["_k", "scorer"]).drop(columns="_k")
    write_table(t, tab_dir, "table6_ranking_by_drug_degree")


ABL_LABEL = {"full": "Full model", "no_fingerprints": "No fingerprints", "placeholder_features": "Random features",
             "no_regularization": "No regularization", "random_negatives_only": "Random negatives only",
             "one_layer": "One layer", "no_contra_edges": "No contraindication edges",
             "no_protein_edges": "No protein edges", "no_ontology_context": "No ontology context",
             "no_indication_edges": "No indication edges"}


def fig6(out_dir, fig_dir, tab_dir):
    d = load(out_dir, "ablation_summary.csv")
    if d is None:
        return
    d = d.rename(columns={d.columns[0]: "variant"}) if "variant" not in d.columns else d
    panels = [("test_auroc_hard", "Hard-negative AUROC"), ("test_auroc_random", "Random-negative AUROC"),
              ("p1_within_auc", "Temporal within-drug AUC"), ("p1_cold_within_auc", "Cold-drug within-drug AUC")]
    panels = [p for p in panels if f"{p[0]}_mean" in d.columns]
    fig, axs = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1.6, 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    y = np.arange(len(d))[::-1]
    for ax, (m, ttl), L in zip(axs, panels, "ABCD"):
        mu, sd = d[f"{m}_mean"].values, d.get(f"{m}_std", pd.Series(np.zeros(len(d)))).fillna(0).values
        full = d.index[d.variant == "full"]
        if len(full):
            ax.axvline(mu[full[0]], color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        for yi, v, e, name in zip(y, mu, sd, d.variant):
            col = CAT[1] if name == "full" else CAT[0]
            ax.plot([v - e, v + e], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=3)
            ax.plot(v, yi, "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=4)
        ax.set_title(ttl, fontsize=7.5)
        clean(ax, "x")
        ax.tick_params(axis="x", labelsize=6.5)
        panel_letter(ax, L)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([ABL_LABEL.get(v, v) for v in d.variant])
    fig.supxlabel("Mean ± s.d. over seeds (dashed line: full model)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, fig_dir, "fig6_ablation")
    show = [("test_auroc_all", "AUROC all"), ("test_auroc_hard", "AUROC hard"), ("test_auroc_random", "AUROC random"),
            ("p1_within_auc", "Temporal within-drug AUC (p1)"), ("p1_cold_within_auc", "Cold-drug within-drug AUC"),
            ("p4_within_auc", "Temporal within-drug AUC (p4)")]
    rows = []
    for _, r in d.iterrows():
        row = {"variant": ABL_LABEL.get(r.variant, r.variant), "n_seeds": int(r.get("n_seeds", np.nan))
               if pd.notna(r.get("n_seeds", np.nan)) else ""}
        for m, nm in show:
            if f"{m}_mean" in d.columns:
                row[nm] = f"{r[f'{m}_mean']:.3f} ± {r.get(f'{m}_std', np.nan):.3f}"
        rows.append(row)
    write_table(pd.DataFrame(rows), tab_dir, "table7_ablation")


def fig7(out_dir, fig_dir, tab_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "t1task_*_summary.csv")))
    if not files:
        print("  skip: no t1task_*_summary.csv")
        return
    frames = []
    for f in files:
        rel = os.path.basename(f)[len("t1task_"):-len("_summary.csv")]
        s = pd.read_csv(f)
        s = s[(s.stratum == "query_bucket") & (s.bucket == "ALL")].copy()
        s["relation"] = rel
        frames.append(s)
    d = pd.concat(frames, ignore_index=True)
    rels = list(pd.unique(d.relation))
    scs = [s for s in ("random", "term_popularity", "ontology_neighbor") if s in set(d.scorer)]
    scs = [s for s in pd.unique(d.scorer) if s not in scs and s != "model"] + scs
    scs = (["model"] if "model" in set(d.scorer) else []) + scs
    metric = "within_query_auc" if "within_query_auc" in d.columns else "within_drug_auc"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, met, ttl, L in ((axs[0], metric, "Within-protein AUC", "A"), (axs[1], "hits@100", "Hits@100", "B")):
        vals = {s: dict(zip(d[d.scorer == s].relation, d[d.scorer == s][met])) for s in scs}
        grouped_bars(ax, [r.replace("_protein", "").replace("bioprocess", "biological\nprocess")
                          .replace("molfunc", "molecular\nfunction").replace("cellcomp", "cellular\ncomponent")
                          for r in rels], scs,
                     {s: {k.replace("_protein", "").replace("bioprocess", "biological\nprocess")
                          .replace("molfunc", "molecular\nfunction").replace("cellcomp", "cellular\ncomponent"): v
                          for k, v in vals[s].items()} for s in scs},
                     fmt="{:.2f}" if met == metric else "{:.3f}")
        if met == metric:
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.set_ylim(0.3, 1.1)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.set_title(ttl)
        ax.set_xlabel("GO relation (edges new in PrimeKG-CL t1)")
        clean(ax)
        panel_letter(ax, L)
    fig.legend(handles=[Patch(facecolor=color_of(s), label=pretty(s)) for s in scs], loc="lower center",
               ncol=len(scs), fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, fig_dir, "fig7_go_tasks")
    cols = [c for c in ("relation", "scorer", "n_queries", "n_proteins", "hits@10", "hits@100", "mrr",
                        metric) if c in d.columns]
    write_table(d[cols].sort_values(["relation", "scorer"]), tab_dir, "table8_go_tasks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--fig-dir", default=os.path.join("output", "manuscript"))
    a = ap.parse_args()
    fdir = os.path.join(a.fig_dir, "figures")
    tdir = os.path.join(a.fig_dir, "tables")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        print(fn.__name__)
        try:
            fn(a.out_dir, fdir, tdir)
        except Exception as e:  # one broken input should not stop the others
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nDone. Figures: {fdir}\n      Tables:  {tdir}")


if __name__ == "__main__":
    main()