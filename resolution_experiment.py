#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resolution_experiment.py — 입력 해상도 스윕 (Qwen3-VL-4B, zero-shot)

배경
----
베이크오프에서 Qwen3-VL-4B는 재질 93% · 색상 95% · 기타 92%인데
**개수 세기만 74%** 였다. 개수 문제는 test의 33.9%를 차지한다.
원본 이미지는 720×960인데 384²(=147k px)로 눌러 넣고 있으므로,
작은 물체를 세는 일이 여기서 가장 먼저 무너진다는 가설을 검증한다.

가설: 해상도를 올리면 개수 정확도가 오른다. 다른 유형은 이미 90%대라 여지가 적다.

설계
----
- 모델 1회 로드 → 해상도마다 processor만 다시 만들어 평가 (로딩 비용 1회)
- 추론은 a/b/c/d 로짓 비교 (베이크오프와 동일, 재현 가능)
- **val_full 507장 전체**로 평가하고, 기존 299장 부분집합 점수도 함께 산출
  → 507로 신뢰구간을 좁히면서 기존 86.3%와도 비교 가능
- 해상도 간 짝지은 McNemar 검정
- 해상도별 결과 즉시 저장 → 중단돼도 재실행 시 이어서

사용법
------
  python resolution_experiment.py --dry-run              # 경로·split 점검 (10초)
  python resolution_experiment.py --sizes 384 512        # 빠른 확인
  python resolution_experiment.py                        # 384/512/640/768 전체
  python resolution_experiment.py --n 299 --batch 1      # 표본 축소 / OOM 대응
"""

from __future__ import annotations

import argparse, copy, gc, inspect, json, math, sys, time, warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_LABEL = "Qwen3-VL-4B"
LETTERS = ["a", "b", "c", "d"]
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


# ── 데이터 (베이크오프와 동일한 split) ────────────────────────────────────────
def qtype(q):
    q = str(q)
    if "몇" in q:   return "개수"
    if "재질" in q: return "재질"
    if "색" in q:   return "색상"
    return "기타"


def build_split(data_dir, seed, val_frac, n_sub):
    tr = pd.read_csv(data_dir / "train.csv")
    tr["qtype"] = tr["question"].map(qtype)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    k = int(len(tr) * val_frac)
    val_full = tr.iloc[perm[:k]].reset_index(drop=True)

    parts = []                                  # 기존 베이크오프의 299장 부분집합
    for qt, g in val_full.groupby("qtype"):
        take = max(1, min(len(g), round(n_sub * len(g) / len(val_full))))
        parts.append(g.sample(take, random_state=seed))
    sub = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sub_ids = set(sub["id"])
    return val_full, sub_ids


def build_mc_prompt(row):
    return (f"{row['question']}\n"
            f"(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")


def build_messages(row):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user",   "content": [{"type": "image"},
                                           {"type": "text", "text": build_mc_prompt(row)}]}]


# ── 모델 로딩 (flash 우회 + auto_map 해석) ───────────────────────────────────
_FLASH = {"flash_attention_2", "flash_attention_3"}


def _scrub_flash(cfg, impl):
    seen = set()
    def walk(c):
        if c is None or id(c) in seen or not hasattr(c, "to_dict"): return
        seen.add(id(c))
        for a in ("_attn_implementation", "attn_implementation"):
            if getattr(c, a, None) in _FLASH:
                try: setattr(c, a, impl)
                except Exception: pass
        try: c._attn_implementation = impl
        except Exception: pass
        for v in list(vars(c).values()):
            if hasattr(v, "to_dict"): walk(v)
    walk(cfg); return cfg


@contextmanager
def no_flash_attention(impl="sdpa"):
    from transformers.modeling_utils import PreTrainedModel as PM
    saved = {}
    def coerce(a, kw):
        a = [impl if (isinstance(v, str) and v in _FLASH) else v for v in a]
        for k, v in list(kw.items()):
            if isinstance(v, str) and v in _FLASH: kw[k] = impl
            elif k == "use_flash_attention_2":     kw[k] = False
        for v in list(a) + list(kw.values()):
            if hasattr(v, "to_dict"): _scrub_flash(v, impl)
        return a, kw
    for name in ("_from_config", "_check_and_adjust_attn_implementation",
                 "_autoset_attn_implementation"):
        raw = inspect.getattr_static(PM, name, None)
        if raw is None: continue
        saved[name] = (raw, name in PM.__dict__)
        is_cm, is_sm = isinstance(raw, classmethod), isinstance(raw, staticmethod)
        fn = raw.__func__ if (is_cm or is_sm) else raw
        def make(fn=fn):
            def w(first, *a, **kw):
                a, kw = coerce(list(a), kw); return fn(first, *a, **kw)
            return w
        w = make()
        setattr(PM, name, classmethod(w) if is_cm else (staticmethod(w) if is_sm else w))
    try:
        yield
    finally:
        for name, (raw, own) in saved.items():
            if own: setattr(PM, name, raw)
            else:
                try: delattr(PM, name)
                except AttributeError: pass


def _resolve_class():
    import transformers
    from transformers import AutoConfig
    try:
        amap = getattr(AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True),
                       "auto_map", None) or {}
    except Exception:
        amap = {}
    for k in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        if k in amap:
            try:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                return get_class_from_dynamic_module(amap[k], MODEL_ID)
            except Exception:
                pass
    for k in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        C = getattr(transformers, k, None)
        if C is not None: return C
    raise RuntimeError("모델 클래스를 찾지 못했습니다")


def load_model(dtype):
    Cls = _resolve_class()
    with no_flash_attention("sdpa"):
        try:
            m = Cls.from_pretrained(MODEL_ID, device_map="auto",
                                    trust_remote_code=True, dtype=dtype)
        except TypeError:
            m = Cls.from_pretrained(MODEL_ID, device_map="auto",
                                    trust_remote_code=True, torch_dtype=dtype)
    return m.eval()


def make_processor(image_size):
    """해상도는 processor의 min/max_pixels로 결정된다 → 해상도마다 새로 만든다."""
    from transformers import AutoProcessor
    px = image_size * image_size
    try:
        p = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=px, max_pixels=px,
                                          trust_remote_code=True)
    except TypeError:
        p = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("  [!] min/max_pixels 미지원 processor — 해상도가 안 바뀔 수 있습니다")
    p.tokenizer.padding_side = "left"
    return p


# ── 추론 ────────────────────────────────────────────────────────────────────
def get_letter_ids(tok):
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False) or tok.encode(" " + L,
                                                                   add_special_tokens=False)
        ids.append(enc[0])
    assert len(set(ids)) == 4, f"토큰 id 중복 {ids}"
    return ids


def evaluate(model, proc, df, image_size, letter_ids, batch, data_dir, max_len):
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def load_image(row):
        img = Image.open(data_dir / row["path"]).convert("RGB")
        img.thumbnail((image_size * 2, image_size * 2), Image.BICUBIC)
        return img

    @torch.no_grad()
    def score(rows):
        try:
            texts = [proc.apply_chat_template(build_messages(r), tokenize=False,
                                              add_generation_prompt=True) for r in rows]
            inputs = proc(text=texts, images=[load_image(r) for r in rows],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=max_len).to(model.device)
            n_tok = int(inputs["input_ids"].shape[1])
            logits = model(**inputs).logits[:, -1, :].float()
            return logits[:, letter_ids].softmax(-1).cpu().numpy(), n_tok
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if len(rows) == 1: raise
            mid = len(rows) // 2
            p1, t1 = score(rows[:mid]); p2, t2 = score(rows[mid:])
            return np.concatenate([p1, p2], 0), max(t1, t2)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    out, toks, t0 = [], [], time.time()
    for s in tqdm(range(0, len(df), batch), desc=f"{image_size}px", unit="배치"):
        p, n = score([df.iloc[i] for i in range(s, min(s + batch, len(df)))])
        out.append(p); toks.append(n)
    probs = np.concatenate(out, 0)
    return probs, time.time() - t0, float(np.mean(toks))


# ── 리포팅 ──────────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(c1, c2):
    n10, n01 = int((c1 & ~c2).sum()), int((~c1 & c2).sum()); n = n10 + n01
    if n == 0: return n10, n01, 1.0
    return n10, n01, math.erfc(math.sqrt(((abs(n10 - n01) - 1) ** 2 / n) / 2))


def draw_chart(recs, val_full, out_dir, gpu, dtype_name, base_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import ft2font
    from matplotlib.ticker import PercentFormatter

    def has_hangul(p):
        try: return ft2font.FT2Font(p).get_char_index(ord("한")) != 0
        except Exception: return False

    known = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
             "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    prefer = ["malgun", "nanum", "apple sd gothic", "noto sans cjk", "gothic"]
    avoid = ["unifont", "sample", "symbol", "emoji"]
    KO = False
    for p in known:
        if Path(p).exists() and has_hangul(p):
            fm.fontManager.addfont(p)
            _n = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = _n
            print(f"[font] {_n}"); KO = True; break
    if not KO:
        hits = []
        for ext in ("ttf", "otf"):
            for p in fm.findSystemFonts(fontext=ext):
                if not has_hangul(p): continue
                try: n = fm.FontProperties(fname=p).get_name()
                except Exception: continue
                low = (n + " " + Path(p).name).lower()
                hits.append((2 if any(b in low for b in avoid) else
                             (0 if any(g in low for g in prefer) else 1), p, n))
        if hits:
            hits.sort(key=lambda h: h[0])
            fm.fontManager.addfont(hits[0][1])
            plt.rcParams["font.family"] = hits[0][2]
            print(f"[font] {hits[0][2]}"); KO = True
    plt.rcParams["axes.unicode_minus"] = False
    T = lambda ko, en: ko if KO else en

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    C_CNT, C_ALL, C_OTH = "#eb6834", "#2a78d6", "#1baf7a"

    def tidy(ax, axis="y"):
        ax.set_facecolor(SURFACE)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        ax.grid(axis=axis, color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True)

    sizes = [r["image_size"] for r in recs]
    x = np.arange(len(sizes))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=SURFACE)
    fig.suptitle(T(f"입력 해상도 스윕  ·  {MODEL_LABEL}  ·  val {recs[0]['n_full']}장  ·  "
                   f"{gpu} / {dtype_name}  ·  학습 없음",
                   f"Resolution sweep · {MODEL_LABEL} · val {recs[0]['n_full']}"),
                 fontsize=13, fontweight="bold", color=INK, y=.985)

    # 1. 전체 vs 개수
    ax = axes[0, 0]; tidy(ax)
    allv = [r["acc_full"] for r in recs]
    cnt = [r["by_type"].get("개수", np.nan) for r in recs]
    ax.plot(x, allv, "-o", color=C_ALL, lw=2, ms=8, zorder=3,
            label=T("전체", "overall"))
    ax.plot(x, cnt, "-o", color=C_CNT, lw=2, ms=8, zorder=3,
            label=T("개수 문제", "counting"))
    for xx, v in zip(x, allv): ax.text(xx, v + .015, f"{v:.1%}", ha="center",
                                       fontsize=9, fontweight="bold", color=INK)
    for xx, v in zip(x, cnt):  ax.text(xx, v - .028, f"{v:.1%}", ha="center",
                                       fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}px" for s in sizes], fontsize=10, color=INK)
    lo = min(min(allv), np.nanmin(cnt)); hi = max(max(allv), np.nanmax(cnt))
    ax.set_ylim(lo - .09, hi + .07)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    ax.set_title(T("해상도별 정확도 — 개수 문제가 오르는가",
                   "Accuracy by resolution"), fontsize=11, fontweight="bold",
                 color=INK, loc="left", pad=10)

    # 2. 유형별
    ax = axes[0, 1]; tidy(ax)
    types = [t for t in ["개수", "재질", "색상", "기타"] if t in recs[0]["by_type"]]
    w = .8 / len(types)
    for i, t in enumerate(types):
        vals = [r["by_type"].get(t, np.nan) for r in recs]
        ax.bar(x + i * w - .4 + w / 2, vals, w * .88, label=t if KO else t,
               color=[C_CNT, C_ALL, C_OTH, "#eda100"][i % 4], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}px" for s in sizes], fontsize=10, color=INK)
    ax.set_ylim(0, 1.26); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=4,
              columnspacing=.9, handlelength=1.1)
    ax.set_title(T("질문 유형별", "By question type"), fontsize=11,
                 fontweight="bold", color=INK, loc="left", pad=10)

    # 3. 비용
    ax = axes[1, 0]; tidy(ax)
    mins = [r["sec_per_sample"] * 5074 / 60 for r in recs]
    ax.bar(x, mins, .45, color=C_OTH, zorder=3)
    for xx, m, r in zip(x, mins, recs):
        ax.text(xx, m + max(mins) * .03,
                T(f"{m:.0f}분\n({r['mean_tokens']:.0f} tok)", f"{m:.0f} min"),
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}px" for s in sizes], fontsize=10, color=INK)
    ax.set_ylim(0, max(mins) * 1.3)
    ax.set_ylabel(T("test 5,074장 예상 (분)", "est. min for 5,074"), fontsize=9, color=INK2)
    ax.set_title(T("추론 비용", "Inference cost"), fontsize=11, fontweight="bold",
                 color=INK, loc="left", pad=10)

    # 4. 384 대비 변화량
    ax = axes[1, 1]; tidy(ax, axis="x")
    base = next(r for r in recs if r["image_size"] == base_size)
    rest = [r for r in recs if r["image_size"] != base_size]
    if rest:
        y = np.arange(len(rest) * 2)[::-1]
        names, vals, cols = [], [], []
        for r in rest:
            names += [T(f"{r['image_size']}px  전체", f"{r['image_size']}px all"),
                      T(f"{r['image_size']}px  개수", f"{r['image_size']}px count")]
            vals += [r["acc_full"] - base["acc_full"],
                     r["by_type"].get("개수", 0) - base["by_type"].get("개수", 0)]
            cols += [C_ALL, C_CNT]
        ax.barh(y, vals, height=.55, color=cols, zorder=3)
        ax.axvline(0, color=INK2, lw=1)
        span = max(abs(v) for v in vals) or .01
        for yy, v in zip(y, vals):
            ax.text(v + (span * .05 if v >= 0 else -span * .05), yy, f"{v:+.1%}p",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9.5, fontweight="bold", color=INK)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.set_xlim(-span * 1.5, span * 1.5)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(T(f"{base_size}px 대비 변화", f"vs {base_size}px"),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)

    plt.tight_layout(rect=[0, 0, 1, .95])
    png = out_dir / "resolution.png"
    plt.savefig(png, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장: {png}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="입력 해상도 스윕 (Qwen3-VL-4B)")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="bakeoff_out")
    ap.add_argument("--sizes", type=int, nargs="+", default=[384, 512, 640, 768])
    ap.add_argument("--n", type=int, default=0, help="0=val 전체(507). 숫자를 주면 그만큼만")
    ap.add_argument("--n-sub", type=int, default=300, help="기존 베이크오프 부분집합 크기")
    ap.add_argument("--batch", type=int, default=0, help="0=VRAM 보고 자동")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "train"):
        if not (data_dir / f).exists():
            sys.exit(f"[중단] {data_dir / f} 없음. --data-dir 확인")

    val_full, sub_ids = build_split(data_dir, args.seed, args.val_frac, args.n_sub)
    if args.n:
        val_full = val_full.head(args.n).reset_index(drop=True)
    gold = val_full["answer"].str.strip().str.lower().values
    in_sub = val_full["id"].isin(sub_ids).values

    print("=" * 72)
    print(f"데이터 {data_dir}\n출력  {out_dir}")
    print(f"val 전체 {len(val_full)}장 · 그중 기존 베이크오프 부분집합 {int(in_sub.sum())}장")
    print("유형 분포:", val_full.qtype.value_counts(normalize=True).mul(100).round(1).to_dict())
    print(f"해상도: {args.sizes}")
    print("=" * 72)
    if args.dry_run:
        print("\n[dry-run] 모델 로드를 건너뜁니다.")
        return

    import torch
    if not torch.cuda.is_available(): sys.exit("[중단] CUDA 없음")
    cc = torch.cuda.get_device_capability(0)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    print(f"\nGPU {gpu} (cc {cc[0]}.{cc[1]}, {vram:.1f}GB) · dtype {str(dtype).split('.')[-1]}")

    todo = [s for s in args.sizes
            if args.force or not (out_dir / f"res_{s}.json").exists()]
    recs = []
    for s in args.sizes:                                   # 저장분 먼저 회수
        p = out_dir / f"res_{s}.json"
        if p.exists() and not args.force:
            recs.append(json.load(open(p, encoding="utf-8")))
            print(f"[SKIP] {s}px — 저장된 결과 ({recs[-1]['acc_full']:.1%})")

    if todo:
        print(f"\n모델 로드 (1회, 첫 실행은 다운로드 포함 10~20분)")
        t = time.time(); model = load_model(dtype)
        print(f"로드 {time.time()-t:.0f}s · VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        for size in todo:
            batch = args.batch or (4 if vram >= 22 else (2 if vram >= 14 else 1))
            if size >= 640: batch = max(1, batch // 2)      # 토큰 수가 급증
            print(f"\n{'─'*72}\n▶ {size}px  (batch {batch})")
            proc = make_processor(size)
            probs, el, mean_tok = evaluate(model, proc, val_full, size,
                                           get_letter_ids(proc.tokenizer),
                                           batch, data_dir, args.max_len)
            pred = np.array([LETTERS[i] for i in probs.argmax(1)])
            corr = (pred == gold)
            by_type = {}
            for qt in val_full.qtype.unique():
                m = (val_full.qtype == qt).values
                by_type[qt] = float(corr[m].mean())
            rec = {"model": MODEL_ID, "label": MODEL_LABEL, "image_size": size,
                   "acc_full": float(corr.mean()), "n_full": int(len(val_full)),
                   "acc_sub": float(corr[in_sub].mean()), "n_sub": int(in_sub.sum()),
                   "by_type": by_type, "correct": corr.tolist(),
                   "pred_dist": {L: float((pred == L).mean()) for L in LETTERS},
                   "sec_per_sample": el / len(val_full), "mean_tokens": mean_tok,
                   "gpu": gpu, "dtype": str(dtype).split(".")[-1], "batch": batch}
            json.dump(rec, open(out_dir / f"res_{size}.json", "w", encoding="utf-8"),
                      ensure_ascii=False)
            recs.append(rec)
            lo, hi = wilson(int(corr.sum()), len(corr))
            print(f"  전체 {rec['acc_full']:.1%} [{lo:.1%},{hi:.1%}] · "
                  f"개수 {by_type.get('개수', float('nan')):.1%} · "
                  f"{rec['sec_per_sample']:.2f}s/장 · 평균 {mean_tok:.0f} tok")
            proc = None; gc.collect(); torch.cuda.empty_cache()
        model = None; gc.collect(); torch.cuda.empty_cache()

    recs.sort(key=lambda r: r["image_size"])
    if not recs: sys.exit("결과 없음")

    print("\n" + "=" * 72)
    print(f"{'해상도':>8} {'전체':>8} {'95% CI':>17} {'개수':>8} {'재질':>7} "
          f"{'색상':>7} {'기타':>7} {'s/장':>7} {'5074장':>8}")
    for r in recs:
        lo, hi = wilson(round(r["acc_full"] * r["n_full"]), r["n_full"])
        b = r["by_type"]
        print(f"{r['image_size']:>6}px {r['acc_full']:>8.1%} "
              f"[{lo:>6.1%},{hi:>6.1%}] {b.get('개수',0):>8.1%} {b.get('재질',0):>7.1%} "
              f"{b.get('색상',0):>7.1%} {b.get('기타',0):>7.1%} "
              f"{r['sec_per_sample']:>7.2f} {r['sec_per_sample']*5074/60:>7.0f}분")
    print(f"\n(참고) 기존 베이크오프와 같은 {recs[0]['n_sub']}장 부분집합 기준")
    for r in recs:
        print(f"  {r['image_size']:>4}px  {r['acc_sub']:.1%}")

    base = recs[0]
    if len(recs) > 1:
        print(f"\nMcNemar — {base['image_size']}px 대비 (짝지은 검정)")
        cb = np.array(base["correct"], bool)
        for r in recs[1:]:
            cr = np.array(r["correct"], bool)
            n10, n01, p = mcnemar(cr, cb)
            print(f"  {r['image_size']:>4}px  {r['acc_full']-base['acc_full']:+.1%}p  "
                  f"(고침 {n10}, 망침 {n01})  p={p:.4f} "
                  f"{'유의' if p < .05 else '무의미'}")
            m = (val_full.qtype == "개수").values
            n10c, n01c, pc = mcnemar(cr[m], cb[m])
            print(f"         개수만: {r['by_type'].get('개수',0)-base['by_type'].get('개수',0):+.1%}p "
                  f"(고침 {n10c}, 망침 {n01c}) p={pc:.4f} "
                  f"{'유의' if pc < .05 else '무의미'}")

    best = max(recs, key=lambda r: r["acc_full"])
    print("\n" + "=" * 72)
    print(f"최고: {best['image_size']}px — 전체 {best['acc_full']:.1%} · "
          f"개수 {best['by_type'].get('개수',0):.1%} · "
          f"5,074장 {best['sec_per_sample']*5074/60:.0f}분")
    if best["image_size"] == base["image_size"]:
        print("→ 해상도를 올려도 나아지지 않았습니다. 가설 기각.")
        print("  개수 문제의 약점은 해상도가 아니라 모델의 계수 능력 자체입니다.")
        print("  다음 레버는 LoRA 파인튜닝(개수 문제 비중을 높인 샘플링 포함).")
    else:
        d = best["by_type"].get("개수", 0) - base["by_type"].get("개수", 0)
        print(f"→ 학습 해상도를 {best['image_size']}px로 확정. 개수 문제 {d:+.1%}p.")
        print(f"  단 학습 시간도 약 {best['sec_per_sample']/base['sec_per_sample']:.1f}배가 됩니다.")
    print("=" * 72)

    json.dump({"model": MODEL_ID, "gpu": gpu, "n_full": recs[0]["n_full"],
               "results": [{k: v for k, v in r.items() if k != "correct"} for r in recs]},
              open(out_dir / "resolution_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"요약 저장: {out_dir / 'resolution_summary.json'}")

    if not args.no_chart:
        draw_chart(recs, val_full, out_dir, recs[0]["gpu"], recs[0]["dtype"],
                   recs[0]["image_size"])


if __name__ == "__main__":
    main()
