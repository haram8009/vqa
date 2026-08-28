#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kanana_bakeoff.py — Kanana-1.5-V-3B 전용 zero-shot 베이크오프 (로컬 실행)

재활용품 VQA 챌린지. 3모델 베이크오프에서 kanana만 계속 실패해서 따로 뺀 스크립트.
학습은 하지 않습니다. 모델을 고르기 위한 zero-shot 측정 전용.

kanana가 실패했던 이유 세 가지와 이 스크립트의 대응:

  1) Unrecognized configuration class ... AutoModelForImageTextToText
     → kanana는 auto_map에 AutoModelForVision2Seq 로만 등록돼 있습니다.
       Auto 클래스를 추측하지 않고 config.auto_map에서 실제 클래스를
       get_class_from_dynamic_module 로 직접 가져옵니다.

  2) No module named 'timm'
     → kanana의 vision encoder가 timm 기반입니다. 누락 패키지 이름을
       에러 메시지에서 파싱해 자동 설치하고 재시도합니다 (timm, einops).

  3) FlashAttention2 has been toggled on, but ... flash_attn not installed
     → flash_attn 을 설치하면 안 됩니다. FlashAttention-2는 Ampere(sm80)
       이상에서만 동작하고 빌드에 10분 이상 걸립니다. config와 모든 하위
       config에서 flash_attention_2 를 걷어내고 sdpa → eager 순으로 내립니다.

결과는 bakeoff_out/result_kanana_v_3b.json 으로 저장되며,
같은 폴더에 Qwen 결과가 있으면 마지막에 3모델 비교 그래프까지 그립니다.

사용법
------
  python kanana_bakeoff.py --dry-run     # 모델 없이 데이터·경로·기준선만 점검 (10초)
  python kanana_bakeoff.py --smoke       # 8장만 돌려 로딩·토큰 id 확인 (fail fast)
  python kanana_bakeoff.py               # val 300장 본 측정
  python kanana_bakeoff.py --n 500 --batch 1 --4bit
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import random
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

KANANA_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
KANANA_KEY = "kanana_v_3b"
KANANA_LABEL = "Kanana-1.5-V-3B"
LETTERS = ["a", "b", "c", "d"]

SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 — 베이크오프 노트북과 동일한 split이어야 결과가 비교 가능합니다.
#          SEED / VAL_FRAC / N 을 바꾸면 기존 Qwen 결과와 나란히 놓을 수 없습니다.
# ─────────────────────────────────────────────────────────────────────────────
def qtype(q) -> str:
    q = str(q)
    if "몇" in q:
        return "개수"
    if "재질" in q:
        return "재질"
    if "색" in q:
        return "색상"
    return "기타"


def build_split(data_dir: Path, seed: int, val_frac: float, n_val: int):
    train_df = pd.read_csv(data_dir / "train.csv")
    train_df["qtype"] = train_df["question"].map(qtype)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_df))
    k = int(len(train_df) * val_frac)
    val_full = train_df.iloc[perm[:k]].reset_index(drop=True)
    fit_df = train_df.iloc[perm[k:]].reset_index(drop=True)

    parts = []                                     # 질문 유형 비율 유지 (층화 추출)
    for qt, g in val_full.groupby("qtype"):
        take = max(1, min(len(g), round(n_val * len(g) / len(val_full))))
        parts.append(g.sample(take, random_state=seed))
    val_df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, fit_df, val_full, val_df


def text_only_baseline(fit_df, val_df):
    """이미지를 전혀 주지 않은 기준선. VLM은 이 값을 넘어야 의미가 있습니다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def to_pairs(df):
        rows = []
        for gid, r in df.iterrows():
            for L in LETTERS:
                rows.append((gid, f"{r['question']} [SEP] {r[L]}",
                             int(str(r["answer"]).strip().lower() == L)))
        return pd.DataFrame(rows, columns=["gid", "text", "y"])

    p_fit, p_val = to_pairs(fit_df), to_pairs(val_df)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, max_features=200_000)
    Xf, Xv = vec.fit_transform(p_fit.text), vec.transform(p_val.text)
    clf = LogisticRegression(max_iter=2000, C=2.0).fit(Xf, p_fit.y)

    probs = clf.predict_proba(Xv)[:, 1].reshape(len(val_df), 4)
    probs = probs / probs.sum(1, keepdims=True)
    pred = np.array([LETTERS[i] for i in probs.argmax(1)])
    gold = val_df["answer"].str.strip().str.lower().values
    return float((pred == gold).mean()), probs


def build_mc_prompt(row) -> str:
    return (
        f"{row['question']}\n"
        f"(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


# ─────────────────────────────────────────────────────────────────────────────
# kanana 로더 — 실패 원인 3가지를 순서대로 방어
# ─────────────────────────────────────────────────────────────────────────────
_DEP_RE = re.compile(r"not found in your environment:\s*([^\.\n]+)")
_MOD_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
# import 이름 → pip 패키지 이름이 다른 경우
_PIP_ALIAS = {"cv2": "opencv-python-headless", "PIL": "pillow", "sklearn": "scikit-learn"}


def _missing_packages(msg: str):
    m = _DEP_RE.search(msg)
    if m:
        pkgs = [p.strip() for p in m.group(1).split(",") if p.strip()]
    else:
        m2 = _MOD_RE.search(msg)
        pkgs = [m2.group(1).split(".")[0]] if m2 else []
    # flash_attn 은 절대 자동 설치하지 않습니다 (아래 주석 참고)
    return [_PIP_ALIAS.get(p, p) for p in pkgs if p != "flash_attn"]


def _pip_install(pkgs):
    print(f"[deps] 누락 패키지 자동 설치: {pkgs}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    importlib.invalidate_caches()


def _clear_remote_modules():
    """실패한 remote code 모듈이 sys.modules에 반쯤 남으면 재시도가 계속 깨집니다."""
    for k in [k for k in list(sys.modules) if k.startswith("transformers_modules")]:
        del sys.modules[k]


def _scrub_flash_attn(cfg, impl):
    """config와 모든 하위 config에서 flash_attention_2를 걷어냅니다."""
    seen = set()

    def walk(c):
        if id(c) in seen or not hasattr(c, "to_dict"):
            return
        seen.add(id(c))
        for attr in ("_attn_implementation", "attn_implementation"):
            if getattr(c, attr, None) in ("flash_attention_2", "flash_attention_3"):
                setattr(c, attr, impl)
        try:
            c._attn_implementation = impl
        except Exception:
            pass
        for v in list(vars(c).values()):
            if hasattr(v, "to_dict"):
                walk(v)

    walk(cfg)
    return cfg


from contextlib import contextmanager

_FLASH = {"flash_attention_2", "flash_attention_3"}


@contextmanager
def no_flash_attention(impl="eager", verbose=True):
    """
    kanana의 modeling.py는 vision tower를 이렇게 하드코딩합니다:

        try:
            self.vision_model = CustomQwen2VLVE._from_config(
                config.vision_config, attn_implementation="flash_attention_2")
        except Exception:
            self.vision_model = CustomQwen2VLVE._from_config(config.vision_config)

    try/except가 있는데도 터지는 이유: 1차 호출이 실패하면서 transformers가
    config.vision_config._attn_implementation 을 'flash_attention_2' 로
    **이미 써버립니다**. 그래서 폴백 호출도 같은 flash를 요청하게 되고,
    이번엔 잡아주는 곳이 없어 ImportError가 밖으로 나옵니다.
    (config를 미리 청소해도 1차 호출이 다시 오염시키므로 소용없습니다.)

    → transformers 진입점 자체에서 flash 요청을 impl로 바꿔치기합니다.
      블록을 벗어나면 원래대로 복구합니다.
    """
    from transformers.modeling_utils import PreTrainedModel as PM

    NAMES = ("_from_config",                            # 모든 버전
             "_check_and_adjust_attn_implementation",   # transformers 4.5x+
             "_autoset_attn_implementation")            # 구버전
    saved, patched = {}, []

    def coerce(a, kw):
        a = [impl if (isinstance(v, str) and v in _FLASH) else v for v in a]
        for k, v in list(kw.items()):
            if isinstance(v, str) and v in _FLASH:
                kw[k] = impl
            elif k == "use_flash_attention_2":
                kw[k] = False
        for v in list(a) + list(kw.values()):
            if hasattr(v, "to_dict"):
                _scrub_flash_attn(v, impl)
        return a, kw

    for name in NAMES:
        raw = importlib.import_module("inspect").getattr_static(PM, name, None)
        if raw is None:
            continue
        saved[name] = (raw, name in PM.__dict__)
        is_cm, is_sm = isinstance(raw, classmethod), isinstance(raw, staticmethod)
        fn = raw.__func__ if (is_cm or is_sm) else raw

        def make(fn=fn):
            def wrapper(first, *a, **kw):
                a, kw = coerce(list(a), kw)
                return fn(first, *a, **kw)
            return wrapper

        w = make()
        setattr(PM, name, classmethod(w) if is_cm else (staticmethod(w) if is_sm else w))
        patched.append(name)

    if verbose and patched:
        print(f"[flash] attn_implementation={impl} 로 강제 (패치: {', '.join(patched)})")
    try:
        yield
    finally:
        for name, (raw, own) in saved.items():
            if own:
                setattr(PM, name, raw)
            else:
                try:
                    delattr(PM, name)
                except AttributeError:
                    pass


def _resolve_kanana_class():
    """auto_map에서 실제 클래스를 직접 가져옵니다 (Auto 등록 우회)."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(KANANA_ID, trust_remote_code=True)
    amap = getattr(cfg, "auto_map", None) or {}
    ref = next((amap[k] for k in ("AutoModelForVision2Seq",
                                  "AutoModelForImageTextToText",
                                  "AutoModelForCausalLM") if k in amap), None)
    if ref is None:
        raise RuntimeError(f"auto_map에 모델 클래스가 없습니다: {list(amap)}")
    print(f"[cls] auto_map 직접 로드 → {ref}")
    return get_class_from_dynamic_module(ref, KANANA_ID), cfg


def load_kanana(dtype, use_4bit, attn_order=("sdpa", "eager"), max_dep_retry=4):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    for dep_attempt in range(max_dep_retry):
        try:
            _clear_remote_modules()
            Cls, base_cfg = _resolve_kanana_class()

            processor = AutoProcessor.from_pretrained(KANANA_ID, trust_remote_code=True)
            processor.tokenizer.padding_side = "left"      # 로짓 스코어링에 필수

            last_err = None
            for impl in attn_order:
                cfg = _scrub_flash_attn(copy.deepcopy(base_cfg), impl)
                base_kw = {"config": cfg, "device_map": "auto", "trust_remote_code": True}
                if use_4bit:
                    base_kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
                else:
                    base_kw["dtype"] = dtype

                # (a) attn_implementation 인자 지원  (b) 미지원 → config만으로
                # 각각에 대해 dtype → torch_dtype 구버전 폴백까지
                variants = []
                for extra in ({"attn_implementation": impl}, {}):
                    for dt_key in ("dtype", "torch_dtype"):
                        kw = dict(base_kw, **extra)
                        if "dtype" in kw and dt_key == "torch_dtype":
                            kw["torch_dtype"] = kw.pop("dtype")
                        variants.append(kw)

                # ★ 이 블록 안에서는 flash_attention_2 요청이 전부 impl로 치환됩니다
                with no_flash_attention(impl):
                    for kw in variants:
                        try:
                            model = Cls.from_pretrained(KANANA_ID, **kw)
                            print(f"[kanana] 로드 성공 (attn_implementation={impl})")
                            return model.eval(), processor
                        except TypeError as e:
                            last_err = e
                            continue                   # 인자 조합 문제 → 다음 변형
                        except ImportError:
                            raise                      # 의존성 → 바깥에서 설치 후 재시도
                        except Exception as e:
                            last_err = e
                            print(f"[kanana] attn={impl} 실패: "
                                  f"{type(e).__name__}: {str(e)[:200]}")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            break                      # 다음 attn 구현으로
            raise last_err if last_err else RuntimeError("kanana 로드 실패")

        except ImportError as e:
            msg = str(e)
            if "flash_attn" in msg or "FlashAttention" in msg:
                # no_flash_attention 패치를 뚫고 나온 경우. flash_attn은 설치하지
                # 않습니다 — Ampere(sm80) 이상 전용이고 빌드가 매우 오래 걸립니다.
                # 같은 에러로 무한 재시도하지 않도록 여기서 바로 중단합니다.
                raise RuntimeError(
                    "flash_attention_2 요청을 끝내 우회하지 못했습니다.\n"
                    "  --attn eager 로 고정해 다시 실행해 보세요.\n"
                    "  그래도 같으면 transformers 버전을 알려주세요 "
                    f"(현재: {__import__('transformers').__version__}).\n"
                    "  flash_attn 설치는 해결책이 아닙니다 (sm80 미만 미지원)."
                ) from e
            pkgs = _missing_packages(msg)
            if not pkgs:
                raise
            _pip_install(pkgs)
    raise RuntimeError("의존성 자동 설치를 반복했지만 kanana 로드에 실패했습니다")


# ─────────────────────────────────────────────────────────────────────────────
# 로짓 비교 추론 — generate() 대신 forward 1회로 a/b/c/d 확률만 비교
# ─────────────────────────────────────────────────────────────────────────────
def get_letter_ids(tokenizer, verbose=True):
    ids, rep = [], []
    for L in LETTERS:
        enc = tokenizer.encode(L, add_special_tokens=False)
        if not enc:
            enc = tokenizer.encode(" " + L, add_special_tokens=False)
        ids.append(enc[0])
        rep.append((L, enc, repr(tokenizer.decode([enc[0]]))))
    if verbose:
        print(f"  {'letter':<8}{'ids':<16}decode(id[0])")
        for L, e, d in rep:
            print(f"  {L:<8}{str(e):<16}{d}")
    assert len(set(ids)) == 4, f"[FAIL] LETTER_IDS 중복! {ids}"
    if verbose:
        print(f"  [OK] 4개 id가 모두 다름: {ids}")
    return ids


def encode_rows(processor, rows, data_dir, image_size, max_len):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    batch = []
    for r in rows:
        img = Image.open(data_dir / r["path"]).convert("RGB")
        img.thumbnail((image_size * 2, image_size * 2), Image.BICUBIC)
        batch.append({
            "image": [img],
            "conv": [{"role": "user", "content": "<image>"},
                     {"role": "user", "content": SYSTEM_INSTRUCT + "\n\n" + build_mc_prompt(r)}],
        })
    return processor.batch_encode_collate(
        batch, padding_side="left", add_generation_prompt=True, max_length=max_len)


def score_rows(model, processor, rows, letter_ids, data_dir, image_size, max_len):
    """OOM이면 배치를 반으로 쪼개 재귀 재시도."""
    import torch
    try:
        inputs = encode_rows(processor, rows, data_dir, image_size, max_len)
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                out = model(**inputs)
            except TypeError:
                drop = {"attention_mask_2d", "generation_config", "max_length", "padding_side"}
                out = model(**{k: v for k, v in inputs.items() if k not in drop})
        logits = out.logits[:, -1, :].float()          # fp16 오버플로 방지
        return logits[:, letter_ids].softmax(-1).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        mid = len(rows) // 2
        return np.concatenate([
            score_rows(model, processor, rows[:mid], letter_ids, data_dir, image_size, max_len),
            score_rows(model, processor, rows[mid:], letter_ids, data_dir, image_size, max_len)], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 결과 병합 & 그래프
# ─────────────────────────────────────────────────────────────────────────────
def setup_korean_font():
    """캐시를 우회해 폰트 파일을 직접 찾고, '한' 글리프 유무로 검증합니다."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import ft2font

    known = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
             r"C:\Windows\Fonts\NanumGothic.ttf", r"C:\Windows\Fonts\gulim.ttc",
             "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    prefer = ["malgun", "nanum", "apple sd gothic", "applegothic", "noto sans cjk",
              "noto sans kr", "source han sans", "pretendard", "gulim", "dotum", "gothic"]
    avoid = ["unifont", "sample", "symbol", "emoji", "wingding", "webding"]

    def ok(p):
        try:
            return ft2font.FT2Font(p).get_char_index(ord("한")) != 0
        except Exception:
            return False

    def register(p):
        fm.fontManager.addfont(p)
        name = fm.FontProperties(fname=p).get_name()
        plt.rcParams["font.family"] = name
        print(f"[font] {name}  ({p})")
        return name

    for p in known:
        if Path(p).exists() and ok(p):
            return register(p)
    hits = []
    for ext in ("ttf", "otf"):
        for p in fm.findSystemFonts(fontext=ext):
            if not ok(p):
                continue
            try:
                name = fm.FontProperties(fname=p).get_name()
            except Exception:
                continue
            low = (name + " " + Path(p).name).lower()
            rank = 2 if any(b in low for b in avoid) else (0 if any(g in low for g in prefer) else 1)
            hits.append((rank, p))
    if hits:
        hits.sort(key=lambda h: h[0])
        return register(hits[0][1])
    return None


def draw_chart(results, val_df, text_only_acc, image_size, out_dir, gpu_name, dtype_name):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ko_font = setup_korean_font()
    KO = ko_font is not None
    plt.rcParams["axes.unicode_minus"] = False
    if not KO:
        print("[font] 한글 폰트를 못 찾아 라벨을 영어로 표시합니다.")

    def T(ko, en):
        return ko if KO else en

    QT_EN = {"개수": "Count", "재질": "Material", "색상": "Color", "기타": "Other"}
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # 검증된 카테고리 팔레트
    CMAP = {r["label"]: SERIES[i % len(SERIES)] for i, r in enumerate(results)}

    def tidy(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        ax.grid(axis="x", color=GRID, lw=.8, zorder=0)
        ax.set_axisbelow(True)

    order = sorted(results, key=lambda r: r["acc"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE)
    fig.suptitle(T(f"Zero-shot 모델 베이크오프  ·  val {len(val_df)}장  ·  {image_size}px  ·  "
                   f"{gpu_name} / {dtype_name}  ·  학습 없음",
                   f"Zero-shot model bakeoff  ·  val {len(val_df)}  ·  {image_size}px  ·  "
                   f"{gpu_name} / {dtype_name}  ·  no training"),
                 fontsize=13, fontweight="bold", color=INK, x=.5, y=.985)

    # 1. 전체 정확도
    ax = axes[0, 0]; tidy(ax)
    y = np.arange(len(order))
    ax.barh(y, [r["acc"] for r in order], height=.42,
            color=[CMAP[r["label"]] for r in order], zorder=3)
    for i, r in enumerate(order):
        ax.text(r["acc"] + .012, i, f"{r['acc']:.1%}", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
    ax.axvline(.25, color=INK2, ls=":", lw=1.4, zorder=4)
    ax.axvline(text_only_acc, color="#e34948", ls="--", lw=1.8, zorder=4)
    ax.text(.25, len(order) - .3, T(" 랜덤 25%", " random 25%"),
            fontsize=8.5, color=INK2, va="bottom")
    ax.text(text_only_acc, len(order) - .3,
            T(f" 텍스트 전용 {text_only_acc:.0%} ← 합격선",
              f" text-only {text_only_acc:.0%} ← pass line"),
            fontsize=8.5, color="#e34948", va="bottom", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in order], fontsize=10, color=INK)
    ax.set_xlim(0, max(.75, max(r["acc"] for r in results) + .14))
    ax.set_ylim(-.6, len(order) - .1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(T("전체 정확도", "Overall accuracy"), fontsize=11,
                 fontweight="bold", color=INK, loc="left", pad=10)

    # 2. 질문 유형별
    ax = axes[0, 1]; tidy(ax)
    types = [t for t in ["개수", "재질", "색상", "기타"] if t in val_df.qtype.unique()]
    share = val_df.qtype.value_counts(normalize=True)
    x = np.arange(len(types)); w = .8 / len(results)
    for i, r in enumerate(results):
        vals = [r["by_type"].get(t, np.nan) for t in types]
        xs = x + i * w - .4 + w / 2
        ax.bar(xs, vals, w * .88, label=r["label"], color=CMAP[r["label"]], zorder=3)
        for xx, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(xx, v + .015, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(text_only_acc, color="#e34948", ls="--", lw=1.4, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{T(t, QT_EN[t])}\n({share.get(t, 0):.0%})" for t in types],
                       fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.26)                       # 범례가 막대를 가리지 않도록 여유
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=.8, zorder=0); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=3,
              columnspacing=1.1, handlelength=1.2)
    ax.set_title(T("질문 유형별 정확도  (괄호 = val 내 비중)",
                   "Accuracy by question type  (share of val)"),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)

    # 3. 예측 분포 진단
    ax = axes[1, 0]; tidy(ax)
    x = np.arange(4); w = .8 / len(results)
    for i, r in enumerate(results):
        ax.bar(x + i * w - .4 + w / 2, [r["pred_dist"][L] for L in LETTERS], w * .88,
               label=r["label"], color=CMAP[r["label"]], zorder=3)
    ax.axhline(.25, color=INK2, ls=":", lw=1.4, zorder=4)
    ax.text(3.45, .255, T("균등 25%", "uniform 25%"), fontsize=8, color=INK2, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"({L})" for L in LETTERS], fontsize=10, color=INK)
    ax.set_ylim(0, max(.42, max(max(r["pred_dist"].values()) for r in results) * 1.25))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=.8, zorder=0); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title(T("예측 분포 진단  ·  한 글자로 쏠리면 프롬프트/토큰 id 문제",
                   "Prediction distribution  ·  skew to one letter = prompt/token-id bug"),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)

    # 4. 추론 속도
    ax = axes[1, 1]; tidy(ax)
    sp = sorted(results, key=lambda r: -r["sec_per_sample"])
    y = np.arange(len(sp)); mins = [r["sec_per_sample"] * 5074 / 60 for r in sp]
    ax.barh(y, mins, height=.42, color=[CMAP[r["label"]] for r in sp], zorder=3)
    for i, (r, m) in enumerate(zip(sp, mins)):
        ax.text(m + max(mins) * .02, i,
                T(f"{m:.0f}분  ({r['sec_per_sample']:.2f}s/장)",
                  f"{m:.0f} min  ({r['sec_per_sample']:.2f}s/img)"),
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in sp], fontsize=10, color=INK)
    ax.set_xlim(0, max(mins) * 1.42); ax.set_ylim(-.6, len(sp) - .1)
    ax.set_xlabel(T("test 5,074장 추론 예상 시간 (분)",
                    "estimated inference time for 5,074 test images (min)"),
                  fontsize=9, color=INK2)
    ax.set_title(T("추론 비용  ·  남는 시간이 곧 학습 예산",
                   "Inference cost  ·  what is left is the training budget"),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)

    plt.tight_layout(rect=[0, 0, 1, .96])
    png = out_dir / "bakeoff.png"
    plt.savefig(png, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그래프 저장: {png}")


def verdict(results, text_only_acc, n_val, out_dir):
    se = math.sqrt(.25 * .75 / n_val)
    print(f"\n표본 {n_val}장 · 표준오차 ≈ ±{se:.1%}p · 무승부 판정폭 ≈ {2*se:.1%}p\n")
    ranked = sorted(results, key=lambda r: -r["acc"])
    for i, r in enumerate(ranked, 1):
        gap = r["acc"] - text_only_acc
        flag = ("PASS" if gap > 2 * se else
                ("FAIL — 이미지 기여 없음" if gap <= 0 else "애매 — 통계적 무의미"))
        print(f"{i}. {r['label']:<22} {r['acc']:.1%}   텍스트전용 대비 {gap:+.1%}p   [{flag}]")

    best = ranked[0]
    tied = [r for r in ranked[1:] if best["acc"] - r["acc"] < 2 * se]
    print(f"\n승자: {best['label']}  ({best['acc']:.1%})")
    if tied:
        print("무승부권:", ", ".join(r["label"] for r in tied),
              "→ 더 빠르고 라이선스가 깨끗한 쪽(Apache 2.0인 Qwen 계열) 권장")
    if best["acc"] - text_only_acc <= 2 * se:
        print("\n[!] 어떤 모델도 텍스트 전용 기준선을 유의하게 못 넘었습니다. 학습으로 넘어가지 마세요.")
        print("    1) LETTER_IDS 4개가 정말 다른가   2) 예측 분포가 한 글자로 쏠렸는가")
        print("    3) padding_side='left' 적용됐는가  4) 이미지가 실제로 들어갔는가")

    with open(out_dir / "bakeoff_summary.json", "w", encoding="utf-8") as f:
        json.dump({"n_val": n_val, "text_only_acc": text_only_acc,
                   "ranking": [{k: r.get(k) for k in
                                ("key", "label", "acc", "by_type", "pred_dist", "sec_per_sample")}
                               for r in ranked]}, f, ensure_ascii=False, indent=2)
    print(f"요약 저장: {out_dir / 'bakeoff_summary.json'}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Kanana-1.5-V-3B 전용 zero-shot 베이크오프")
    ap.add_argument("--data-dir", default=".", help="train.csv / train 폴더가 있는 경로")
    ap.add_argument("--out-dir", default="bakeoff_out", help="결과 저장 폴더")
    ap.add_argument("--n", type=int, default=300, help="val 표본 수")
    ap.add_argument("--batch", type=int, default=0, help="배치 크기 (0=VRAM 보고 자동)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", help="4bit 양자화 (VRAM 부족 시)")
    ap.add_argument("--attn", default="", choices=["", "sdpa", "eager"],
                    help="어텐션 구현 고정 (기본: sdpa → eager 순으로 시도)")
    ap.add_argument("--smoke", action="store_true", help="8장만 돌려 빠르게 점검")
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 데이터·경로·기준선만 점검")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = 8 if args.smoke else args.n

    print("=" * 70)
    print(f"데이터  : {data_dir}")
    print(f"출력    : {out_dir}")
    print("=" * 70)

    for f in ("train.csv", "train"):
        if not (data_dir / f).exists():
            sys.exit(f"[중단] {data_dir / f} 가 없습니다. --data-dir 를 확인하세요.")

    random.seed(args.seed); np.random.seed(args.seed)

    # 1) split — 노트북과 동일해야 Qwen 결과와 비교 가능
    train_df, fit_df, val_full, val_df = build_split(data_dir, args.seed, args.val_frac, n_val)
    print(f"\ntrain {len(train_df)} / val_full {len(val_full)} / 측정 대상 {len(val_df)}장")
    print(val_df.qtype.value_counts(normalize=True).mul(100).round(1).to_dict())

    # 2) 합격선
    text_only_acc, text_probs = text_only_baseline(fit_df, val_df)
    np.save(out_dir / "text_only_probs.npy", text_probs)
    print(f"\n텍스트 전용 기준선 : {text_only_acc:.1%}   ← kanana는 이 값을 넘어야 의미 있음")
    print(f"랜덤              : 25.0%")

    if args.dry_run:
        print("\n[dry-run] 모델 로드를 건너뜁니다. 경로·split·기준선 점검 완료.")
        return

    # 3) 환경
    import torch
    if not torch.cuda.is_available():
        sys.exit("[중단] CUDA를 찾을 수 없습니다.")
    cc = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16      # T4 등은 fp16
    batch = args.batch or (4 if vram >= 22 else (2 if vram >= 14 else 1))
    print(f"\nGPU   : {gpu_name} (cc {cc[0]}.{cc[1]}, {vram:.1f}GB)")
    print(f"dtype : {str(dtype).split('.')[-1]}   배치: {batch}   4bit: {args.use_4bit}")

    # 4) 로드
    print(f"\n{'='*70}\nkanana 로드 (첫 실행은 모델 다운로드로 10~20분 걸릴 수 있습니다)\n{'='*70}")
    attn_order = (args.attn,) if args.attn else ("sdpa", "eager")
    t0 = time.time()
    model, processor = load_kanana(dtype, args.use_4bit, attn_order=attn_order)
    print(f"로드 완료 ({time.time()-t0:.0f}s) · VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    print("\n[토큰 id 검증]")
    letter_ids = get_letter_ids(processor.tokenizer)

    # 5) 평가
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    all_probs, t0 = [], time.time()
    for s in tqdm(range(0, len(val_df), batch), desc="kanana", unit="batch"):
        rows = [val_df.iloc[i] for i in range(s, min(s + batch, len(val_df)))]
        all_probs.append(score_rows(model, processor, rows, letter_ids,
                                    data_dir, args.image_size, args.max_len))
    elapsed = time.time() - t0

    probs = np.concatenate(all_probs, 0)
    pred = np.array([LETTERS[i] for i in probs.argmax(1)])
    gold = val_df["answer"].str.strip().str.lower().values
    acc = float((pred == gold).mean())

    by_type = {}
    for qt in val_df.qtype.unique():
        m = (val_df.qtype == qt).values
        by_type[qt] = float((pred[m] == gold[m]).mean())

    res = {"key": KANANA_KEY, "label": KANANA_LABEL, "id": KANANA_ID,
           "acc": acc, "by_type": by_type,
           "pred_dist": {L: float((pred == L).mean()) for L in LETTERS},
           "sec_per_sample": elapsed / len(val_df), "n_val": len(val_df),
           "gpu": gpu_name, "dtype": str(dtype).split(".")[-1],
           "use_4bit": args.use_4bit, "image_size": args.image_size,
           "probs": probs.tolist()}

    print(f"\n{'='*70}")
    print(f"kanana 정확도 : {acc:.1%}   (텍스트 전용 {text_only_acc:.1%} / 랜덤 25.0%)")
    print(f"속도          : {res['sec_per_sample']:.2f}s/장  ·  test 5,074장 추정 "
          f"{res['sec_per_sample']*5074/60:.0f}분")
    print(f"예측 분포     : {  {k: round(v,3) for k,v in res['pred_dist'].items()} }")
    print("=" * 70)

    if args.smoke:
        print("\n[smoke] 8장 점검이라 정확도는 의미 없습니다. 로딩·토큰 id·추론이 되면 성공.")
        print("        이제 인자 없이 다시 실행하세요:  python kanana_bakeoff.py")
        return

    with open(out_dir / f"result_{KANANA_KEY}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"저장: {out_dir / f'result_{KANANA_KEY}.json'}")

    # 6) 기존 Qwen 결과와 병합
    results = []
    for p in sorted(out_dir.glob("result_*.json")):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if r.get("n_val") != len(val_df):
            print(f"[!] {p.name} 은 표본 수가 달라({r.get('n_val')}) 비교에서 제외합니다.")
            continue
        results.append(r)

    if len(results) < 2:
        print("\n비교할 다른 모델 결과가 없습니다. 베이크오프 노트북을 먼저 돌리세요.")
        return
    order = {"qwen3vl_4b": 0, "kanana_v_3b": 1, "qwen25vl_3b": 2}
    results.sort(key=lambda r: order.get(r["key"], 99))

    if not args.no_chart:
        draw_chart(results, val_df, text_only_acc, args.image_size, out_dir,
                   gpu_name, str(dtype).split(".")[-1])
    verdict(results, text_only_acc, len(val_df), out_dir)


if __name__ == "__main__":
    main()
