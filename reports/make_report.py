"""Build reports/report.pdf: a short report with only the essentials of the project, written so that
every group member can explain it. Every number is computed from results/ when the report is built.

Contents: problem and data -> pre-processing and evaluation protocol -> the two classifiers (CNN, RNN)
and their training strategy -> iterations (what we changed and why) -> results -> comparison with the
literature -> what the RNN looks at -> conclusions -> appendix with likely questions and short answers.

Usage:
    python reports/make_report.py            # writes report.tex + figures/ and compiles report.pdf
    python reports/make_report.py --no-pdf   # only the .tex

The PDF is compiled with tectonic (conda install -c conda-forge tectonic) or latexmk -xelatex.
Needs data/processed/ and results/<model>/ (oof.csv, cv_summary.json, history_fold*.csv; the fold
models of results/rnn_att for the attention analysis).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CLASSES, STATES, load_segmentation, load_split  # noqa: E402
from src.datasets import CNNWindows, fold_masks, load_processed, mel_band_stats  # noqa: E402
from src.evaluate import RESULTS_DIR, cv_report, load_oof, patient_scores, tune_thresholds  # noqa: E402
from src.metrics import weighted_accuracy  # noqa: E402

OUT = ROOT / "reports"
FIG = OUT / "figures"
INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"cnn": "#eb6834", "rnn": "#2a78d6", "mlp": "#7a7974"}

# the two classifiers of the project, and the iterations tried on each (folder -> label)
FINAL = {"cnn": "CNN", "rnn": "RNN"}
ITERATIONS = {
    "cnn": [("cnn", "base"), ("cnn_plateau", "+ descida da lr"), ("cnn_wa", "+ paragem pela WA"),
            ("cnn_plateau_wa", "+ descida da lr + paragem pela WA")],
    "rnn": [("rnn", "base"), ("rnn_att", "+ atenção"), ("rnn_plateau", "+ descida da lr"),
            ("rnn_wa", "+ paragem pela WA"), ("rnn_plateau_wa", "+ descida da lr + paragem pela WA")],
}
# every model trained in the project: folder -> (name, role, parameters)
ALL_MODELS = {
    "cnn": ("CNN", "classificador 1", 106_083),
    "rnn": ("RNN (BiGRU)", "classificador 2", 115_587),
    "mlp": ("MLP", "primeira abordagem", 59_651),
    "resnet50": ("ResNet50", "literatura (4.º lugar)", 23_507_907),
    "dual_resnet50": ("Dual ResNet50", "literatura (4.º lugar)", 47_015_814),
    "dual_resnet50_imagenet": ("Dual ResNet50 ImageNet", "literatura (4.º lugar)", 47_015_814),
    "hearheart_arch": ("HearHeart, só a rede", "literatura (1.º lugar)", 37_827),
    "hearheart": ("HearHeart, receita completa", "literatura (1.º lugar)", 38_027),
}
MIN_GAIN = 0.02     # an iteration is adopted only if it gains more than this and wins in >= 4 of 5 folds


# ---------------------------------------------------------------- markdown -> LaTeX

UNICODE = {"→": r"$\rightarrow$", "≈": r"$\approx$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
           "−": r"$-$", "±": r"$\pm$", "Δ": r"$\Delta$", "§": r"\S{}", "÷": r"$\div$", "·": r"$\cdot$"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(SPECIAL.get(ch, UNICODE.get(ch, ch)) for ch in s)


def inline(s):
    """Inline markdown: `code`, $math$, [text](url), **bold**, *italic*."""
    keep = []

    def hold(m):
        if m.group(1) is not None:
            keep.append(r"\texttt{" + esc(m.group(1)) + "}")
        elif m.group(2) is not None:
            keep.append("$" + m.group(2) + "$")
        else:
            keep.append(r"\href{" + m.group(4) + "}{" + _emph(esc(m.group(3))) + "}")
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"`([^`]+)`|\$([^$]+)\$|\[([^\]]+)\]\(([^)]+)\)", hold, s)
    s = re.sub(r'"([^"]+)"', "\u201c\\1\u201d", s)          # "texto" -> curly quotes
    s = _emph(esc(s))
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], s)


def _emph(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)


def md_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    head_row = " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\"
    lines = [r"{\small", r"\begin{xltabular}{\linewidth}{" + "L" * len(head) + "}", r"\toprule", head_row,
             r"\midrule", r"\endfirsthead", r"\toprule", head_row, r"\midrule", r"\endhead"]
    lines += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{xltabular}}"]
    return "\n".join(lines)


def md(text):
    """Block-level markdown -> LaTeX: # section, ## subsection, tables, lists, paragraphs."""
    out, para, items, kind, table = [], [], [], None, []

    def flush():
        nonlocal para, items, kind, table
        if para:
            out.append(inline(" ".join(para)) + "\n")
        if items:
            env = "itemize" if kind == "ul" else "enumerate"
            out.append(f"\\begin{{{env}}}\n" + "\n".join(r"\item " + inline(i) for i in items) + f"\n\\end{{{env}}}\n")
        if table:
            out.append(md_table(table))
        para, items, kind, table = [], [], None, []

    for line in text.strip("\n").splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            cmd = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection*"}.get(level, r"\paragraph")
            out.append(f"{cmd}{{{inline(s.lstrip('#').strip())}}}\n")
        elif s.startswith("|"):
            if not table:
                flush()
            table.append(s)
        elif re.match(r"^[-*] ", s) or re.match(r"^\d+\. ", s):
            k = "ul" if s[0] in "-*" else "ol"
            if para or table or (items and k != kind):
                flush()
            kind = k
            items.append(re.sub(r"^([-*]|\d+\.) ", "", s))
        elif items and line.startswith(" "):
            items[-1] += " " + s
        else:
            if items or table:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


def figure(path, width=r"\linewidth", caption=None):
    cap = f"\\caption{{{inline(caption)}}}\n" if caption else ""
    return (f"\\begin{{figure}}[H]\\centering\n\\includegraphics[width={width},height=0.4\\textheight,"
            f"keepaspectratio]{{figures/{path.name}}}\n{cap}\\end{{figure}}\n")


def f3(x):
    return f"{x:.3f}"


def pm(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


# ---------------------------------------------------------------- numbers

def evaluate(d, folder):
    idx, oof = load_oof(folder)
    pat = patient_scores(d.windows.loc[idx], oof)
    table, _, rep = cv_report(pat)
    cfg = json.loads((RESULTS_DIR / folder / "cv_summary.json").read_text())["cfg"]
    hist = {int(p.stem.split("fold")[1].split("_")[0]): pd.read_csv(p)
            for p in sorted((RESULTS_DIR / folder).glob("history_fold?.csv"))}
    y = pat.murmur
    auc = {c: roc_auc_score(y == c, pat[c]) for c in ["Present", "Unknown"]}
    return dict(pat=pat, wa=table["WA (tuned)"], rep=rep, cfg=cfg, hist=hist, auc=auc)


def best_epoch(h, cfg):
    return int(h.val_wa.idxmax()) if cfg.get("monitor", "loss") == "wa" else int(h.val_loss.idxmin())


def dataset_numbers(d):
    rec = d.recordings
    pat = rec.groupby("patient_id").murmur.first()
    pres = rec[rec.murmur == "Present"]
    dev = d.windows[d.windows.fold >= 0].groupby("patient_id").murmur.first()
    return dict(n_pat=len(pat), n_rec=len(rec), n_win=len(d.windows),
                share={c: (pat == c).mean() for c in CLASSES},
                no_murmur=(pres.rec_label != "Present").mean(),
                base={c: weighted_accuracy(dev, [c] * len(dev)) for c in ["Present", "Absent"]},
                n_dev=len(dev), n_test=d.windows[d.windows.fold == -1].patient_id.nunique())


def attention_lift(d):
    """Attention mass / time share per heart-cycle phase for the RNN with attention (validation windows
    with >= 80 % annotated frames, 150 Present + 150 Absent windows per fold, model of that fold)."""
    import torch
    from src.rnn import load_runs
    runs, _ = load_runs(RESULTS_DIR / "rnn_att")
    rows = []
    for r in runs:
        w = d.windows[d.windows.fold == r["fold"]]
        sel = pd.concat([w[w.rec_label == c].sample(150, random_state=0) for c in ["Present", "Absent"]])
        ds = CNNWindows(d, d.windows.index.isin(sel.index), r["band_stats"])
        m = r["model"].cpu().eval()
        with torch.no_grad():
            _, att = m(torch.stack([ds[i][0] for i in range(len(ds))]), return_weights=True)
        att = att[:, :, 0].numpy()
        for i, row in enumerate(ds.rows.itertuples()):
            seg = load_segmentation(row.recording)
            t = row.start_s + np.arange(401) * 0.01
            state = np.zeros(401, int)
            for s in seg.itertuples():
                state[(t >= s.start) & (t < s.end)] = s.state
            if (state == 0).mean() > 0.2:
                continue
            for k in (1, 2, 3, 4):
                if (state == k).any():
                    rows.append({"rec": row.rec_label, "fase": STATES[k], "att": att[i, state == k].sum(),
                                 "tempo": (state == k).mean()})
    g = pd.DataFrame(rows).groupby(["fase", "rec"])[["att", "tempo"]].mean()
    return (g.att / g.tempo).unstack("rec")


# ---------------------------------------------------------------- figures

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)


def fig_inputs(d):
    tr, va, _ = fold_masks(d.windows, val_fold=0)
    ds = CNNWindows(d, va, mel_band_stats(d, tr))
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))
    for c, ax in zip(CLASSES, axes):
        i = int(np.flatnonzero(ds.rows.rec_label == c)[0])
        ax.imshow(ds[i][0][0], origin="lower", aspect="auto", cmap="magma", extent=[0, 4, 0, 40])
        ax.set_title(c, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("tempo (s)", fontsize=9, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
    axes[0].set_ylabel("banda mel", fontsize=9, color=INK2)
    fig.tight_layout()
    p = FIG / "inputs.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_wa(res, base):
    names = ["cnn", "rnn", "mlp"]
    labels = {"cnn": "CNN", "rnn": "RNN", "mlp": "MLP (1.ª abordagem)"}
    fig, ax = plt.subplots(figsize=(7, 2.4))
    y = np.arange(len(names))[::-1]
    m = [res[n]["wa"].mean() for n in names]
    s = [res[n]["wa"].std() for n in names]
    ax.barh(y, m, height=0.5, color=[COLORS[n] for n in names], zorder=2)
    ax.errorbar(m, y, xerr=s, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    for yi, mi, si in zip(y, m, s):
        ax.text(mi + si + 0.01, yi, f"{mi:.3f} ± {si:.3f}", va="center", fontsize=9, color=INK)
    for c, v in base.items():
        ax.axvline(v, color=INK2, ls="--", lw=1, zorder=1)
        left = c == min(base, key=base.get)              # lower baseline: label to the left of its line
        ax.text(v - 0.005 if left else v + 0.005, y.max() + 0.45, f"sempre {c}: {v:.2f}",
                ha="right" if left else "left", fontsize=8, color=INK2)
    ax.set_yticks(y, [labels[n] for n in names], fontsize=10)
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("WA (média ± desvio padrão nos 5 folds)", fontsize=9, color=INK2)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = FIG / "results_wa.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_curves(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for (f, name), ax in zip(FINAL.items(), axes):
        for h in res[f]["hist"].values():
            ax.plot(h.epoch + 1, h.train_loss, color=INK2, lw=1, alpha=0.35)
            ax.plot(h.epoch + 1, h.val_loss, color=COLORS[f], lw=1.8)
            b = best_epoch(h, res[f]["cfg"])
            ax.plot(b + 1, h.val_loss[b], "o", ms=6, color=COLORS[f], mec="white", mew=1.5)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("época", fontsize=9, color=INK2)
        style(ax)
    axes[0].set_ylabel("cross-entropy ponderada", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=INK2, alpha=0.35, label="treino (um traço por fold)"),
               plt.Line2D([], [], color=INK, lw=2, label="validação"),
               plt.Line2D([], [], marker="o", ls="none", color=INK, label="época guardada")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = FIG / "results_curves.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def tex_confusion(res):
    blocks = []
    for f, name in FINAL.items():
        m = res[f]["rep"]["confusion"]
        rows = [f"\\textit{{{c}}} & " + " & ".join(str(v) for v in m[i]) + r" \\" for i, c in enumerate(CLASSES)]
        blocks.append(r"\begin{minipage}[t]{0.42\linewidth}\centering\footnotesize"
                      + f"\\textbf{{{name}}} (WA agregada {res[f]['rep']['weighted_accuracy']:.3f})\\\\[2pt]\n"
                      + r"\begin{tabular}{lccc}\toprule previsto $\downarrow$ / verdadeiro $\rightarrow$ & P & U & A \\ \midrule"
                      + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}\end{minipage}")
    return r"\begin{center}" + "\n\\hfill\n".join(blocks) + r"\end{center}"


# ---------------------------------------------------------------- sections

def sec_summary(res, ds):
    c, r = res["cnn"], res["rnn"]
    return md(f"""
# Resumo

O objetivo é dizer, para cada criança, se tem um sopro cardíaco (*Present*), se não tem (*Absent*) ou se a
gravação não permite decidir (*Unknown*), a partir dos sons do coração (dataset CirCor DigiScope). Implementámos
os dois classificadores pedidos de entre MLP, CNN e RNN: uma **CNN** e uma **RNN**, que recebem a mesma entrada
(o mel-espectrograma de janelas de 4 s) e são treinadas e avaliadas da mesma forma.

| | CNN | RNN |
|---|---|---|
| Como lê o som | procura padrões locais na "imagem" do som | lê o som por ordem, frame a frame |
| WA (validação cruzada, 5 folds) | **{pm(c['wa'])}** | **{pm(r['wa'])}** |
| Recall *Present* / *Unknown* / *Absent* | {' / '.join(f"{c['rep']['recall'][k]:.2f}" for k in CLASSES)} | {' / '.join(f"{r['rep']['recall'][k]:.2f}" for k in CLASSES)} |

Os dois modelos **empatam** e estão muito acima do valor mínimo a bater (prever sempre *Present*:
{ds['base']['Present']:.2f}). Também comparámos com dois modelos do PhysioNet Challenge 2022, incluindo o vencedor,
reimplementados nas mesmas condições: nenhum ganha à nossa CNN.
""")


def sec_data(ds):
    s = ds["share"]
    return md(f"""
# O problema e os dados

- **Dados:** {ds['n_pat']} pacientes (0 a 21 anos) e {ds['n_rec']} gravações a 4 kHz, de 5 a 65 s. Em geral há uma
  gravação por foco de auscultação (aórtico, pulmonar, tricúspide, mitral).
- **O rótulo é do paciente**, não da gravação: {s['Absent']:.0%} *Absent*, {s['Present']:.0%} *Present*,
  {s['Unknown']:.0%} *Unknown*. Temos de dar **uma previsão por paciente**.
- **O sopro pode ouvir-se só nalguns focos:** {ds['no_murmur']:.0%} das gravações de pacientes *Present* não têm
  sopro audível. Usamos o campo `Murmur locations` para dar a cada gravação o seu próprio rótulo no treino.
- **Os sopros são quase todos sistólicos** (entre os sons S1 e S2) e fracos, e na sístole há ~5 dB mais energia
  acima de 150 Hz quando há sopro. O modelo tem de ver ciclos cardíacos completos com boa resolução no tempo.
- ***Unknown* é sobretudo "má gravação":** as gravações *Unknown* são as menos periódicas e as mais ruidosas.

**A métrica** é a *weighted accuracy* (WA) por paciente: cada *Present* conta 5 vezes, cada *Unknown* 3 e cada
*Absent* 1, porque falhar um sopro é pior do que um falso alarme. Prever sempre *Absent* dá só
{ds['base']['Absent']:.2f}; prever sempre *Present* dá {ds['base']['Present']:.2f}, **o valor mínimo a bater**.
""")


def sec_preprocessing(d, ds, p_inputs):
    split = load_split()
    ct = pd.crosstab(split.fold.replace({-1: "teste"}).astype(str), split.murmur)[CLASSES]
    rows = "\n".join(f"| {k} | " + " | ".join(str(v) for v in row) + f" | {row.sum()} |" for k, row in ct.iterrows())
    return md(f"""
# Pré-processamento e protocolo de avaliação

| Decisão | Porquê |
|---|---|
| Filtro passa-banda 25–1000 Hz e normalização da amplitude de cada gravação | abaixo de 25 Hz há movimento; acima de 1 kHz quase só ruído; o volume varia entre estetoscópios |
| Janelas de 4 s com passo de 2 s ({ds['n_win']} janelas) | as gravações têm durações muito diferentes; 4 s têm 3 a 7 ciclos cardíacos completos |
| Mel-espectrograma 40 bandas × 401 frames (20–1000 Hz, um frame a cada 10 ms) | 10 ms chegam para separar S1, sístole e S2, que é onde está o sopro |
| Sem normalização por janela | o nível absoluto em dB ajuda a reconhecer as gravações ruidosas (*Unknown*) |
| Normalização por banda mel só com os dados de treino de cada fold | evita fuga de informação da validação/teste para o treino |
| Rótulo por gravação no treino, avaliação sempre por paciente | {ds['no_murmur']:.0%} das gravações de pacientes *Present* não têm sopro |

**De janelas a pacientes.** A rede dá probabilidades para cada janela. Fazemos a média por gravação; o paciente
fica com o **máximo** de P(*Present*) e de P(*Unknown*) entre as suas gravações (basta um foco com sopro). A decisão
é hierárquica, como no Challenge: *Present* se P(*Present*) ≥ t1; senão *Unknown* se P(*Unknown*) ≥ t2; senão
*Absent*. Os limiares t1 e t2 são escolhidos para maximizar a WA.

**Divisão dos dados por paciente**, estratificada pelo rótulo: {ds['n_test']} pacientes (20 %) formam o **conjunto
de teste**, nunca usado para decidir nada; os restantes {ds['n_dev']} formam **5 folds** de validação cruzada. As 70
crianças vistas nas duas campanhas de recolha ficam sempre no mesmo fold. Cada modelo é treinado 5 vezes (uma por
fold) e reportamos a média ± desvio padrão da WA nos 5 folds. **Os limiares de cada fold são afinados só nos outros
4 folds**, para a estimativa não ser otimista.

| Fold | Present | Unknown | Absent | Total |
|---|---|---|---|---|
{rows}
""") + figure(p_inputs, caption="Entrada dos dois modelos: uma janela de 4 s de cada classe (mel-espectrograma normalizado por banda).")


def sec_models(res):
    c, r = res["cnn"]["cfg"], res["rnn"]["cfg"]
    return md(f"""
# Os dois classificadores

## CNN 2D

Trata o mel-espectrograma como uma imagem. Os filtros convolucionais deslizam pelo tempo e pela frequência e
aprendem padrões locais (por exemplo "energia entre S1 e S2").

- 4 blocos `Conv 3×3 (stride 2) → BatchNorm → ReLU` com 16 → 32 → 64 → 128 mapas. O stride 2 reduz a imagem para
  metade em cada bloco (40 × 401 → 3 × 26), por isso cada neurónio do último bloco vê ~310 ms: S1 + sístole + S2.
- *Global average pooling* (média de cada mapa) e classificador `Dropout 0.3 → 128 → 64 → Dropout 0.3 → 3`.
- 106 mil parâmetros.

## RNN (GRU bidirecional)

Lê a mesma janela como uma **sequência de 401 frames** (10 ms cada), cada um com as 40 bandas mel. Uma GRU guarda
uma memória do que já leu; sendo bidirecional, em cada instante sabe o que veio antes e o que vem depois, ou seja,
em que fase do ciclo cardíaco está.

- 2 camadas GRU bidirecionais com 64 unidades por direção, dropout 0.3 entre camadas.
- As saídas dos 401 frames fazem-se a média e passam por `Dropout 0.3 → Linear → 3`.
- 116 mil parâmetros, quase os mesmos da CNN, para a comparação ser justa.

A **MLP** (99 features resumidas por janela → 256 → 128 → 3) foi a primeira abordagem (WA
{pm(res['mlp']['wa'])}). Foi substituída pela RNN porque as médias e desvios padrão da janela perdem o **momento**
em que a energia aparece, que é o que define um sopro.

## Estratégia de treino (igual nos dois)

| | |
|---|---|
| Função de perda | cross-entropy com **pesos de classe** inversamente proporcionais à frequência (uma janela *Unknown* pesa ~17× uma *Absent*) |
| Desequilíbrio | pesos de classe na perda + limiares de decisão afinados para a WA |
| Otimizador | Adam, learning rate {c['lr']:g} constante, weight decay {c['weight_decay']:g} |
| Mini-batch / épocas | {c['batch_size']} janelas; máximo {c['max_epochs']} épocas |
| Paragem (*early stopping*) | quando a perda de validação não melhora durante {c['patience']} épocas; ficam os pesos da melhor época |
| Regularização | dropout 0.3, weight decay, BatchNorm (CNN) |
| Aumento de dados | deslocamento aleatório da janela no tempo e *SpecAugment* (uma máscara de frequência e uma de tempo por batch) |
| Transfer learning | não (testado numa ResNet50 com pesos ImageNet: não funcionou, ver secção 6) |
| Hardware | uma GPU A100 do Deucalion; ~10 min por modelo para os 5 folds |

**Como escolhemos os hiperparâmetros.** Com ~600 pacientes de treino por fold, procurar muitas combinações na
validação cruzada levaria a escolher a que teve sorte nos folds. Por isso fixámos valores padrão antes de treinar
(Adam 1e-3, batch 128, dropout 0.3). Os dados só escolhem o número de épocas (early stopping), os limiares
(afinados de forma encadeada) e as iterações que ficam (secção seguinte).
""")


def sec_iterations(res, p_curves):
    rows = []
    for model, its in ITERATIONS.items():
        base = res[its[0][0]]["wa"]
        for f, label in its:
            wa = res[f]["wa"]
            gain, wins = wa.mean() - base.mean(), int((wa > base).sum())
            if f == its[0][0]:
                dec = "referência"
            else:
                dec = "adotada" if gain > MIN_GAIN and wins >= 4 else "rejeitada"
            rows.append(f"| {FINAL[model]} | {label} | {pm(wa)} | {'—' if f == its[0][0] else f'{gain:+.3f}'} | "
                        f"{'—' if f == its[0][0] else f'{wins}/5'} | {dec} |")
    return md(f"""
# Iterações: o que mudámos e porquê

Cada alteração partiu de uma observação. **Regra fixada antes de ver os resultados:** uma alteração só fica se
aumentar a WA média em mais de {MIN_GAIN} (cerca de um desvio padrão entre folds) e ganhar em pelo menos 4 dos 5
folds. Caso contrário fica a versão mais simples.

| Alteração | Porquê |
|---|---|
| **Atenção** na RNN (média ponderada dos frames em vez da média simples) | deixar a rede focar os instantes com sopro |
| **Descida da learning rate** (÷10 após 3 épocas sem melhorar) | as curvas de validação eram ruidosas; no modelo vencedor do Challenge a descida da lr foi decisiva |
| **Paragem pela WA** por paciente em vez da perda por janela | a perda por janela não está alinhada com a métrica |

| Modelo | Versão | WA | Δ vs base | Folds a ganhar | Decisão |
|---|---|---|---|---|---|
{chr(10).join(rows)}

**Como interpretámos:**
- **Atenção:** não muda a WA, mas mostra para onde a rede olha (secção 7).
- **Descida da lr:** os modelos convergem em 5 a 15 épocas, antes de a descida fazer diferença. Na RNN a lr só
  desce depois da melhor época.
- **Paragem pela WA:** a WA de um fold é calculada com ~150 pacientes (só ~11 *Unknown*) e oscila ±0.03 a 0.05 de
  época para época. Escolher o máximo dessa curva escolhe a época com mais sorte: na RNN o desvio padrão entre folds
  duplica. A perda, calculada sobre milhares de janelas, é um critério mais estável.
- **Nenhuma alteração passou a regra:** os modelos finais são as versões base.
""") + figure(p_curves, caption="Curvas de treino dos modelos finais: a perda de validação estabiliza ao fim de 5 a 15 épocas e o early stopping guarda a melhor época.")


def sec_results(res, ds, p_wa):
    c, r = res["cnn"], res["rnn"]
    folds = "\n".join(f"| {k} | {c['wa'][k]:.3f} | {r['wa'][k]:.3f} | {r['wa'][k] - c['wa'][k]:+.3f} |" for k in c["wa"].index)
    rows = "\n".join(
        f"| {name} | {pm(res[f]['wa'])} | {res[f]['rep']['weighted_accuracy']:.3f} | "
        + " | ".join(f"{res[f]['rep']['recall'][k]:.2f}" for k in CLASSES)
        + f" | {res[f]['auc']['Present']:.3f} | {res[f]['auc']['Unknown']:.3f} |" for f, name in FINAL.items())
    t = {name: tune_thresholds(res[f]["pat"])[0] for f, name in FINAL.items()}
    return md(f"""
# Resultados

Validação cruzada com {ds['n_dev']} pacientes de desenvolvimento. *WA agregada* junta os 5 folds numa só matriz de
confusão. Recall e AUC são métricas adicionais (a AUC mede se o modelo ordena bem os casos, sem depender dos
limiares).

| Modelo | WA (média ± dp) | WA agregada | Recall P | Recall U | Recall A | AUC Present | AUC Unknown |
|---|---|---|---|---|---|---|---|
{rows}

| Fold | CNN | RNN | RNN − CNN |
|---|---|---|---|
{folds}
""") + tex_confusion(res) + figure(p_wa, width=r"0.8\linewidth") + md(f"""
**Leitura.**
- **CNN e RNN empatam:** a diferença ({r['wa'].mean() - c['wa'].mean():+.3f}) é menor do que a variação entre folds.
  Modelar a sequência inteira não acrescenta aos padrões locais da CNN, que já vê S1 + sístole + S2.
- **A RNN distingue melhor os sopros** (AUC *Present* {r['auc']['Present']:.3f} contra {c['auc']['Present']:.3f})
  e **a CNN reconhece melhor o *Unknown*** (recall {c['rep']['recall']['Unknown']:.2f} contra
  {r['rep']['recall']['Unknown']:.2f}).
- ***Unknown* é a classe mais difícil** para os dois. É a mais pequena, é definida por "o médico não conseguiu
  decidir" (sem um padrão acústico próprio) e vale só ~11 % do peso total da métrica, por isso os limiares tendem a
  sacrificá-la a favor das outras classes.
- **Limiares finais** (afinados em todos os folds, usados no teste): CNN t1 = {t['CNN'][0]:.3f}, t2 =
  {t['CNN'][1]:.3f}; RNN t1 = {t['RNN'][0]:.3f}, t2 = {t['RNN'][1]:.3f}.
- **Teste:** os {ds['n_test']} pacientes de teste ainda não foram avaliados. Avaliam-se uma única vez, com os modelos
  já fechados (média dos 5 modelos dos folds e os limiares acima).
""")


def mean_best_epoch(r):
    return np.mean([best_epoch(h, r["cfg"]) + 1 for h in r["hist"].values()])


def sec_literature(res):
    base = res["cnn"]["wa"]
    hh_rule = json.loads((RESULTS_DIR / "hearheart" / "cv_summary.json").read_text())["hearheart_rule"]["wa_mean"]
    rows = []
    for f, (name, role, p) in ALL_MODELS.items():
        wa = res[f]["wa"]
        params = f"{p / 1e6:.1f} M" if p >= 1e6 else f"{p / 1e3:.0f} mil"
        rows.append(f"| {name} | {role} | {params} | {pm(wa)} | {'—' if f == 'cnn' else f'{int((wa > base).sum())}/5'} | "
                    f"{res[f]['rep']['recall']['Present']:.2f} / {res[f]['rep']['recall']['Unknown']:.2f} |")
    return md(f"""
# Comparação com a literatura

Para saber se os nossos modelos estão ao nível do estado da arte, reimplementámos dois modelos do PhysioNet
Challenge 2022 e treinámo-los **com os nossos folds e a nossa avaliação**, para a comparação ser justa. (Os números
publicados no Challenge vêm de um conjunto de teste escondido e não se comparam diretamente.)

- **ResNet50** (Walker et al., 4.º lugar): uma CNN muito profunda, 220× maior do que a nossa. Três versões: só a
  rede; dividida em dois classificadores binários (*Present* vs resto, *Unknown* vs resto); e com pesos pré-treinados
  no ImageNet.
- **HearHeart** (Lu et al., 1.º lugar), reimplementado a partir do código oficial: uma CNN ainda mais pequena do que
  a nossa. Duas versões: só a rede com o nosso treino; e a receita completa deles (segmentos de 15 s, frequências até
  2 kHz, descida da lr, pesos de classe 5/3/1).

| Modelo | Papel | Parâmetros | WA | Folds a ganhar à CNN | Recall P / U |
|---|---|---|---|---|---|
{chr(10).join(rows)}

**O que aprendemos:**
- **Mais capacidade não ajuda:** as ResNet50 sobreajustam mais cedo (melhor época
  {mean_best_epoch(res['resnet50']):.0f} contra {mean_best_epoch(res['cnn']):.0f} na CNN, em média) e não ganham. O pré-treino no ImageNet falhou: filtros aprendidos em fotografias não servem para estes espectrogramas.
- **A nossa CNN está ao nível do vencedor do Challenge:** a receita completa do HearHeart fica muito perto da CNN
  (e chega a {hh_rule:.3f} com a regra de decisão deles), mas não a ultrapassa. O ganho deles vem da estratégia de treino e
  não da rede: com o nosso treino, a rede deles fica pior.
""")


def sec_attention(lift):
    pt = {"S1": "S1", "systole": "sístole", "S2": "S2", "diastole": "diástole"}
    rows = "\n".join(f"| {pt[fase]} | {lift.loc[fase, 'Absent']:.2f} | {lift.loc[fase, 'Present']:.2f} |"
                     for fase in ["S1", "systole", "S2", "diastole"])
    return md(f"""
# Para onde olha a RNN?

Na versão com atenção, a RNN dá um peso a cada frame. Comparámos esses pesos com as segmentações manuais do ciclo
cardíaco (usadas só para interpretar, não no treino). *Lift* = fração da atenção numa fase ÷ fração do tempo dessa
fase; acima de 1, a fase recebe mais atenção do que a sua duração justificaria.

| Fase | Lift, gravações sem sopro | Lift, gravações com sopro |
|---|---|---|
{rows}

A rede ancora-se no **S1**, o som mais forte e regular. Nas gravações **com sopro** dá mais atenção à **sístole**
({lift.loc['systole', 'Absent']:.2f} → {lift.loc['systole', 'Present']:.2f}) e menos ao S2: olha mais para onde está o
sopro quando ele existe. Numa GRU bidirecional, o estado no S1 já resume o que vem a seguir, por isso focar o S1 não
significa ignorar a sístole.
""")


def sec_conclusions(res):
    return md(f"""
# Conclusões, limitações e trabalho futuro

1. **Os dois classificadores funcionam e empatam:** CNN {pm(res['cnn']['wa'])} e RNN {pm(res['rnn']['wa'])}, muito
   acima do mínimo de 0.50. Para detetar o sopro, os padrões locais da CNN chegam; ler a sequência inteira (RNN) não
   acrescenta.
2. **O que mais pesou foram as decisões sobre os dados:** rótulos por gravação, janelas com ciclos completos,
   agregação pelo máximo por paciente e limiares afinados para a métrica.
3. **Os nossos modelos estão ao nível do estado da arte:** nem uma rede 220× maior nem a receita do vencedor do
   Challenge os ultrapassam nas mesmas condições.
4. **Limitações:** a classe *Unknown* continua difícil (rótulo ruidoso, poucos exemplos); cada configuração foi
   treinada com uma só seed; não houve pesquisa de hiperparâmetros (de propósito); e o early stopping usa o fold
   onde depois se mede a WA, o que torna os números de validação ligeiramente otimistas.
5. **Trabalho futuro:** várias seeds por configuração, um critério de paragem mais estável, e usar as segmentações
   do ciclo cardíaco como tarefa auxiliar no treino.
""")


def sec_qa(res):
    c, r = res["cnn"], res["rnn"]
    return md(f"""
# Apêndice — perguntas prováveis na apresentação

**Porque é que o rótulo por gravação é diferente do rótulo do paciente?** Porque o sopro pode ouvir-se só nalguns
focos. Se todas as gravações de um paciente *Present* fossem *Present*, ensinávamos a rede a dizer "sopro" em
gravações sem sopro. Na avaliação usamos sempre o rótulo do paciente.

**Como passam de janelas para um paciente?** Média das janelas por gravação; o paciente fica com o máximo de
P(*Present*) e de P(*Unknown*) entre gravações (basta um foco com sopro); depois a regra com dois limiares.

**Porque usam limiares em vez da classe mais provável?** Porque a métrica pesa as classes de forma diferente
(5/3/1). Os limiares ajustam a decisão à métrica. São afinados noutros folds, para não serem otimistas.

**Como trataram o desequilíbrio?** Pesos de classe na perda (inverso da frequência) e limiares afinados para a WA.

**Como evitaram fuga de informação (*data leakage*)?** Divisão por paciente (as crianças vistas duas vezes ficam
no mesmo fold), normalização calculada só com o treino de cada fold, e o conjunto de teste nunca usado para decidir.

**Como escolheram os hiperparâmetros?** Valores padrão fixados antes de treinar. Os dados escolhem só as épocas
(early stopping), os limiares e as iterações, com uma regra fixada à partida. Procurar muitas combinações com ~600
pacientes por fold levaria a escolher a que teve sorte.

**Porque é que a CNN e a RNN dão quase o mesmo?** O sopro é um padrão local (entre S1 e S2) e cada neurónio do
último bloco da CNN já vê ~310 ms, o suficiente. A RNN vê a janela inteira, mas isso não acrescenta informação útil.

**Porque é que o *Unknown* é tão difícil?** Não é um som, é "o médico não conseguiu decidir" (ruído, choro,
movimento). Tem poucos exemplos, o rótulo é inconsistente (há crianças com rótulos diferentes nas duas visitas) e
vale só ~11 % do peso da métrica.

**Porque não usaram uma rede maior ou pré-treinada?** Testámos: a ResNet50 (220× maior) sobreajusta e não ganha, e o
pré-treino no ImageNet falhou.

**Quais são os resultados finais?** CNN {pm(c['wa'])} e RNN {pm(r['wa'])} de WA em validação cruzada; o teste final
faz-se uma única vez, no fim.
""")


# ---------------------------------------------------------------- document

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[portuguese]{babel}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,tabularx,xltabular,array,float,xcolor,enumitem,caption}
\usepackage[htt]{hyphenat}
\usepackage[hidelinks]{hyperref}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{3em}
\setlist{itemsep=2pt,topsep=3pt}
\setcounter{secnumdepth}{1}
"""


def build(pdf=True):
    FIG.mkdir(parents=True, exist_ok=True)
    for f in FIG.glob("*.png"):
        f.unlink()
    d = load_processed()
    res = {f: evaluate(d, f) for f in set(ALL_MODELS) | {f for its in ITERATIONS.values() for f, _ in its}}
    ds = dataset_numbers(d)
    print("computing attention analysis ...", flush=True)
    lift = attention_lift(d)
    body = [
        sec_summary(res, ds), sec_data(ds), sec_preprocessing(d, ds, fig_inputs(d)), sec_models(res),
        sec_iterations(res, fig_curves(res)), sec_results(res, ds, fig_wa(res, ds["base"])), sec_literature(res),
        sec_attention(lift), sec_conclusions(res), r"\newpage", sec_qa(res),
    ]
    tex = (PREAMBLE + "\\title{Deteção de sopros cardíacos em fonocardiogramas\\\\"
           "\\large CNN e RNN sobre o CirCor DigiScope --- Machine Learning II}\n"
           f"\\date{{{date.today():%d/%m/%Y}}}\n\\author{{Gerado por \\texttt{{reports/make\\_report.py}}}}\n"
           "\\begin{document}\n\\maketitle\n\\tableofcontents\n\\newpage\n" + "\n".join(body) + "\n\\end{document}\n")
    (OUT / "report.tex").write_text(tex)
    print("written", OUT / "report.tex")
    if pdf:
        compile_pdf()


def compile_pdf():
    tectonic = shutil.which("tectonic") or str(Path(sys.prefix) / "bin" / "tectonic")
    if Path(tectonic).exists():
        cmd = [tectonic, "report.tex"]
    elif shutil.which("latexmk"):
        cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "report.tex"]
    else:
        sys.exit("no LaTeX engine found: conda install -c conda-forge tectonic (report.tex was written)")
    subprocess.run(cmd, cwd=OUT, check=True)
    print("written", OUT / "report.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    build(pdf=not ap.parse_args().no_pdf)
