#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lora_train.py — Qwen3-VL-4B LoRA 파인튜닝 (4지선다 VQA)

설계에서 신경 쓴 것
------------------
1) **정답 토큰만 loss** — 베이스라인의 `labels = input_ids.clone()`은 시스템 프롬프트·
   질문·선택지·이미지 토큰까지 전부 학습 대상으로 삼는다. 정답 한 글자의 비중이 1%도
   안 되는 셈. 여기서는 프롬프트 전체를 -100으로 마스킹하고 **정답 토큰만 남긴다.**
   런타임에 "끝 n_ans 토큰을 디코딩하면 정말 정답인가"를 assert로 검증한다.

2) **학습 목표와 추론 방식을 일치시킴** — 추론은 "생성 프롬프트 직후 위치에서
   a/b/c/d 확률 비교"다. 그래서 학습도 정확히 그 위치의 정답 토큰 확률만 올린다.
   eos도 붙이지 않는다. 학습과 추론이 같은 것을 본다.

3) **선택지 순서 셔플 증강** — 매 epoch 선택지를 섞고 정답 문자를 따라 옮긴다.
   모델이 위치를 외우는 것을 막는다 (베이크오프에서 kanana가 (d)일 때만 52%로
   무너진 그 편향).

4) **학습 전/후를 같은 실행에서 측정** — zero-shot을 먼저 재고 학습 후 다시 잰다.
   외부 숫자와 비교할 필요 없이 순수한 학습 효과가 나온다.

5) **시간 캡 + 주기적 체크포인트** — 중단돼도 어댑터가 남는다.

평가
----
val_full 507장 전체 + 기존 베이크오프와 같은 299장 부분집합 점수를 함께 산출.
추론은 a/b/c/d 로짓 비교 (베이크오프·해상도 실험과 동일).

사용법
------
  python lora_train.py --dry-run                  # 데이터·마스킹 검증만 (GPU 불필요 구간)
  python lora_train.py --max-train 200            # 파이프라인 점검 (10~20분)
  python lora_train.py                            # 전체 학습
  python lora_train.py --image-size 640 --rank 16 --epochs 2
  python lora_train.py --eval-only --adapter bakeoff_out/lora_qwen3vl/final
"""

from __future__ import annotations

import argparse, gc, inspect, json, math, os, random, sys, time, warnings
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


# ── 데이터 ───────────────────────────────────────────────────────────────────
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
    fit_df   = tr.iloc[perm[k:]].reset_index(drop=True)
    parts = []
    for qt, g in val_full.groupby("qtype"):
        take = max(1, min(len(g), round(n_sub * len(g) / len(val_full))))
        parts.append(g.sample(take, random_state=seed))
    sub_ids = set(pd.concat(parts)["id"])
    return fit_df, val_full, sub_ids


def build_mc_prompt(q, a, b, c, d):
    return (f"{q}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")


def row_options(row, rng=None):
    """선택지 순서 셔플 증강. rng=None이면 원래 순서 그대로."""
    opts = [str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])]
    gold_i = LETTERS.index(str(row["answer"]).strip().lower())
    if rng is None:
        return opts, LETTERS[gold_i]
    order = rng.permutation(4)                     # 새 자리 i 에 옛 선택지 order[i]
    new_opts = [opts[j] for j in order]
    new_gold = LETTERS[int(np.where(order == gold_i)[0][0])]
    return new_opts, new_gold


def build_messages(q, opts, image_placeholder=True):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user",   "content": [{"type": "image"},
                                           {"type": "text",
                                            "text": build_mc_prompt(q, *opts)}]}]


def load_image(path, image_size):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    img.thumbnail((image_size * 2, image_size * 2), Image.BICUBIC)
    return img


# ── flash 우회 + 클래스 해석 (다른 스크립트와 동일) ──────────────────────────
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


def make_processor(image_size):
    from transformers import AutoProcessor
    px = image_size * image_size
    try:
        p = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=px, max_pixels=px,
                                          trust_remote_code=True)
    except TypeError:
        p = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return p


def load_base_model(dtype, use_4bit):
    import torch
    from transformers import BitsAndBytesConfig
    Cls = _resolve_class()
    kw = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    else:
        kw["dtype"] = dtype
    with no_flash_attention("sdpa"):
        try:
            return Cls.from_pretrained(MODEL_ID, **kw)
        except TypeError:
            kw["torch_dtype"] = kw.pop("dtype", dtype)
            return Cls.from_pretrained(MODEL_ID, **kw)


# ── 학습 샘플 인코딩: 정답 토큰만 loss ───────────────────────────────────────
def encode_train_sample(proc, img, question, opts, gold_letter, max_len, strict=True):
    """
    반환: input_ids/attention_mask/pixel 관련 + labels
    labels는 **마지막 정답 토큰만** 남기고 전부 -100.

    프롬프트는 add_generation_prompt=True 로 만든다 → 추론 때와 정확히 같은 위치에서
    정답 토큰을 예측하도록 학습된다. eos는 붙이지 않는다.
    """
    import torch
    prompt_text = proc.apply_chat_template(
        build_messages(question, opts), tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + gold_letter

    enc = proc(text=[full_text], images=[img], return_tensors="pt",
               truncation=True, max_length=max_len)
    ids = enc["input_ids"][0]

    tok = proc.tokenizer
    ans_ids = tok.encode(gold_letter, add_special_tokens=False)
    n_ans = max(1, len(ans_ids))

    if strict:                                   # 끝 n_ans 토큰이 정말 정답인가
        tail = tok.decode(ids[-n_ans:]).strip().lower()
        if tail != gold_letter:
            raise ValueError(
                f"라벨 정렬 실패: 끝 {n_ans}토큰이 {tail!r} (기대 {gold_letter!r}). "
                f"토크나이저가 경계를 넘어 병합했을 수 있습니다.")

    labels = ids.clone()
    labels[:-n_ans] = -100                       # ★ 프롬프트 전체 마스킹
    enc["labels"] = labels.unsqueeze(0)
    return enc


def verify_masking(proc, fit_df, data_dir, image_size, max_len, n=5):
    """마스킹이 의도대로 됐는지 눈으로 확인 + assert."""
    print("\n[라벨 마스킹 검증]")
    tok = proc.tokenizer
    for i in range(min(n, len(fit_df))):
        row = fit_df.iloc[i]
        opts, gold = row_options(row)
        img = load_image(data_dir / row["path"], image_size)
        enc = encode_train_sample(proc, img, row["question"], opts, gold, max_len)
        lab = enc["labels"][0]
        kept = (lab != -100).sum().item()
        total = lab.numel()
        kept_txt = tok.decode(lab[lab != -100]).strip()
        print(f"  샘플{i}: 전체 {total:5d}토큰 중 학습 대상 {kept}개 "
              f"({kept/total:.3%}) = {kept_txt!r}  (정답 {gold!r})")
        assert kept == 1 or kept_txt.lower() == gold, "마스킹 불일치"
    print("  → 프롬프트·이미지 토큰은 전부 -100, 정답 토큰만 학습 대상. OK")


# ── 평가 (로짓 비교) ─────────────────────────────────────────────────────────
def get_letter_ids(tok):
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False) or tok.encode(" " + L,
                                                                   add_special_tokens=False)
        ids.append(enc[0])
    assert len(set(ids)) == 4, f"토큰 id 중복 {ids}"
    return ids


def evaluate(model, proc, df, data_dir, image_size, max_len, batch, tag=""):
    import torch
    letter_ids = get_letter_ids(proc.tokenizer)
    proc.tokenizer.padding_side = "left"          # 로짓 스코어링에 필수

    @torch.no_grad()
    def score(rows):
        try:
            texts = [proc.apply_chat_template(
                build_messages(r["question"], row_options(r)[0]),
                tokenize=False, add_generation_prompt=True) for r in rows]
            imgs = [load_image(data_dir / r["path"], image_size) for r in rows]
            inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len).to(model.device)
            logits = model(**inputs).logits[:, -1, :].float()
            return logits[:, letter_ids].softmax(-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if len(rows) == 1: raise
            m = len(rows) // 2
            return np.concatenate([score(rows[:m]), score(rows[m:])], 0)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    was_training = model.training
    model.eval()
    out, t0 = [], time.time()
    for s in tqdm(range(0, len(df), batch), desc=f"eval {tag}", unit="배치"):
        out.append(score([df.iloc[i] for i in range(s, min(s + batch, len(df)))]))
    if was_training: model.train()
    probs = np.concatenate(out, 0)
    pred = np.array([LETTERS[i] for i in probs.argmax(1)])
    gold = df["answer"].str.strip().str.lower().values
    corr = (pred == gold)
    by_type = {}
    for qt in df.qtype.unique():
        m = (df.qtype == qt).values
        by_type[qt] = float(corr[m].mean())
    return {"acc": float(corr.mean()), "by_type": by_type, "correct": corr.tolist(),
            "pred_dist": {L: float((pred == L).mean()) for L in LETTERS},
            "sec_per_sample": (time.time() - t0) / len(df)}


def wilson(k, n, z=1.96):
    if n == 0: return 0., 0.
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(tag, res, sub_mask, prev=None):
    n = len(res["correct"]); k = int(sum(res["correct"]))
    lo, hi = wilson(k, n)
    sub = np.array(res["correct"])[sub_mask]
    line = (f"{tag:<12} 전체 {res['acc']:.1%} [{lo:.1%},{hi:.1%}] (n={n}) · "
            f"299장 부분집합 {sub.mean():.1%}")
    if prev is not None:
        line += f" · 이전 대비 {res['acc']-prev:+.1%}p"
    print(line)
    print("   " + " · ".join(f"{t} {v:.1%}" for t, v in sorted(res["by_type"].items())))
    print("   예측분포 " + " ".join(f"{L}:{v:.0%}" for L, v in res["pred_dist"].items()))


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL-4B LoRA 파인튜닝")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="bakeoff_out")
    ap.add_argument("--run-name", default="lora_qwen3vl")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lm-only", action="store_true",
                    help="언어모델 쪽에만 LoRA (기본: vision 포함 전체 q/k/v/o/gate/up/down)")
    ap.add_argument("--bf16", action="store_true", help="4bit 대신 bf16 (VRAM 여유 시)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="gradient checkpointing 끄기 — 약 30%% 빨라지지만 VRAM을 더 씀. "
                         "OOM 나면 이 플래그를 빼세요")
    ap.add_argument("--no-shuffle-options", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="0=전체 4566")
    ap.add_argument("--time-cap-min", type=int, default=0, help="0=제한 없음")
    ap.add_argument("--save-every", type=int, default=200, help="optimizer step 기준")
    ap.add_argument("--eval-batch", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--n-sub", type=int, default=300)
    ap.add_argument("--resume-adapter", default="",
                    help="기존 어댑터에서 이어서 학습 (rank/alpha/target이 같아야 함)")
    ap.add_argument("--start-epoch", type=int, default=0,
                    help="이어 학습 시 데이터 순서·선택지 셔플 시드 오프셋. "
                         "1에폭 끝낸 뒤 이어붙이면 1을 준다")
    ap.add_argument("--warmup-ratio", type=float, default=-1.0,
                    help="-1=자동 (신규 0.03, 이어학습 0.0)")
    ap.add_argument("--skip-zeroshot", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter", default="", help="eval-only 시 불러올 어댑터 경로")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir  = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    run_dir  = out_dir / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "train"):
        if not (data_dir / f).exists():
            sys.exit(f"[중단] {data_dir/f} 없음. --data-dir 확인")

    random.seed(args.seed); np.random.seed(args.seed)
    fit_df, val_full, sub_ids = build_split(data_dir, args.seed, args.val_frac, args.n_sub)
    sub_mask = val_full["id"].isin(sub_ids).values
    if args.max_train:
        fit_df = fit_df.head(args.max_train).reset_index(drop=True)

    print("=" * 74)
    print(f"모델   {MODEL_ID}")
    print(f"데이터 {data_dir}   출력 {run_dir}")
    print(f"학습 {len(fit_df)}장 / 검증 {len(val_full)}장 (그중 부분집합 {int(sub_mask.sum())}장)")
    print(f"해상도 {args.image_size} · epoch {args.epochs} · lr {args.lr} · "
          f"grad_accum {args.grad_accum} · LoRA r{args.rank}/a{args.alpha}")
    print(f"선택지 셔플 {not args.no_shuffle_options} · "
          f"{'bf16' if args.bf16 else '4bit NF4'}"
          + (f" · 시간 캡 {args.time_cap_min}분" if args.time_cap_min else ""))
    print("=" * 74)

    if args.dry_run:
        print("\n[dry-run] 모델 없이 split만 확인했습니다.")
        print("유형 분포(val):",
              val_full.qtype.value_counts(normalize=True).mul(100).round(1).to_dict())
        return

    import torch
    if not torch.cuda.is_available(): sys.exit("[중단] CUDA 없음")
    cc   = torch.cuda.get_device_capability(0)
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    eval_batch = args.eval_batch or (4 if vram >= 22 else (2 if vram >= 14 else 1))
    if args.image_size >= 640: eval_batch = max(1, eval_batch // 2)
    print(f"\nGPU {gpu} (cc {cc[0]}.{cc[1]}, {vram:.1f}GB) · dtype {str(dtype).split('.')[-1]} "
          f"· eval batch {eval_batch}")

    proc = make_processor(args.image_size)
    model = load_base_model(dtype, use_4bit=not args.bf16)
    print(f"베이스 로드 완료 · VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    verify_masking(proc, fit_df, data_dir, args.image_size, args.max_len)

    metrics = {"model": MODEL_ID, "gpu": gpu, "args": vars(args),
               "n_fit": len(fit_df), "n_val": len(val_full)}

    # ── eval-only ────────────────────────────────────────────────────────────
    if args.eval_only:
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            print(f"어댑터 로드: {args.adapter}")
        res = evaluate(model, proc, val_full, data_dir, args.image_size,
                       args.max_len, eval_batch, tag="adapter")
        report("eval-only", res, sub_mask)
        json.dump(res, open(run_dir / "eval_only.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        return

    # ── 1) 학습 전 zero-shot ─────────────────────────────────────────────────
    before = None
    if not args.skip_zeroshot:
        print(f"\n{'─'*74}\n[1/3] 학습 전 zero-shot 측정")
        before = evaluate(model, proc, val_full, data_dir, args.image_size,
                          args.max_len, eval_batch, tag="before")
        report("학습 전", before, sub_mask)
        metrics["before"] = {k: v for k, v in before.items() if k != "correct"}
        json.dump(metrics, open(run_dir / "metrics.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # ── 2) LoRA 부착 ─────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not args.bf16:
        model = prepare_model_for_kbit_training(model)
    if args.no_grad_ckpt:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        print("gradient checkpointing: 끔 (빠르지만 VRAM 더 씀)")
    else:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing: 켬")
    model.enable_input_require_grads()

    base_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    if args.lm_only:
        targets = [n for n, _ in model.named_modules()
                   if any(n.endswith(t) for t in base_targets)
                   and ("visual" not in n and "vision" not in n)]
        print(f"\nLoRA 대상: 언어모델 전용 {len(targets)}개 모듈")
    else:
        targets = base_targets
        print(f"\nLoRA 대상: {base_targets} (vision 포함)")

    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"이어 학습: {args.resume_adapter} 에서 어댑터 로드")
        print("  ⚠️ rank/alpha/target이 원 학습과 같아야 합니다. 다르면 shape 불일치로 실패합니다.")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
            target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # ── 3) 학습 루프 ─────────────────────────────────────────────────────────
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    n_steps = args.epochs * math.ceil(len(fit_df) / args.grad_accum)
    warmup = args.warmup_ratio if args.warmup_ratio >= 0 else (0.0 if args.resume_adapter else 0.03)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, int(n_steps * warmup), n_steps)
    print(f"스케줄러: {n_steps} step · warmup {warmup:.0%}")

    if args.resume_adapter:                       # Adam 모멘트 복원 (있으면)
        tp = Path(args.resume_adapter) / "trainer_state.pt"
        if tp.exists():
            try:
                st = torch.load(tp, map_location="cpu", weights_only=False)
                optim.load_state_dict(st["optimizer"])
                print(f"  옵티마이저 상태 복원 (이전 {st.get('step','?')} step)")
            except Exception as e:
                print(f"  [!] 옵티마이저 상태 복원 실패({type(e).__name__}) — 새로 시작합니다")
        else:
            print("  [!] trainer_state.pt 없음 — Adam 모멘트 없이 시작 (초기 몇 step이 불안정)")

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    print(f"\n{'─'*74}\n[2/3] 학습 — {args.epochs} epoch × {len(fit_df)}장 "
          f"= 최대 {n_steps} step")
    print("   (첫 20샘플 속도를 보고 총 소요를 아래에 추정합니다)")
    model.train()
    t_start = time.time(); step = 0; skipped = 0; running = 0.0
    stop = False
    loss_log = []

    for ep_i in range(args.epochs):
        ep = args.start_epoch + ep_i              # 데이터 순서·셔플 시드용 (이어학습 대응)
        rng = np.random.RandomState(args.seed + ep)
        order = rng.permutation(len(fit_df))
        shuf_rng = None if args.no_shuffle_options else np.random.RandomState(args.seed * 100 + ep)
        bar = tqdm(order, desc=f"epoch {ep+1} (이번 실행 {ep_i+1}/{args.epochs})", unit="샘플")
        for it, idx in enumerate(bar, start=1):
            row = fit_df.iloc[idx]
            try:
                opts, gold = row_options(row, shuf_rng)
                img = load_image(data_dir / row["path"], args.image_size)
                enc = encode_train_sample(proc, img, row["question"], opts, gold,
                                          args.max_len)
                enc = {k: (v.to(model.device) if torch.is_tensor(v) else v)
                       for k, v in enc.items()}
                loss = model(**enc).loss / args.grad_accum
                loss.backward()
                running += loss.item()
            except torch.cuda.OutOfMemoryError:
                optim.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache(); skipped += 1; continue
            except ValueError as e:                 # 라벨 정렬 실패 샘플은 건너뜀
                skipped += 1
                if skipped <= 3: print(f"\n  [skip] {e}")
                continue

            if it == 20:
                sps = (time.time() - t_start) / 20
                tot = sps * len(fit_df) * args.epochs / 60
                print(f"\n   [속도] {sps:.2f}s/샘플 → 총 예상 {tot:.0f}분"
                      + (f" (시간 캡 {args.time_cap_min}분에서 잘림)"
                         if args.time_cap_min and tot > args.time_cap_min else ""))

            if it % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                step += 1
                loss_log.append(running)
                bar.set_postfix({"loss": f"{running:.3f}", "step": step})
                running = 0.0

                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(run_dir / "checkpoint")
                    torch.save({"optimizer": optim.state_dict(), "step": step,
                                "epoch": ep}, run_dir / "checkpoint" / "trainer_state.pt")
                    json.dump({"step": step, "loss_log": loss_log},
                              open(run_dir / "train_state.json", "w"), ensure_ascii=False)

                if args.time_cap_min and (time.time() - t_start) / 60 >= args.time_cap_min:
                    print(f"\n시간 캡 {args.time_cap_min}분 도달 → 학습 중단 (step {step})")
                    stop = True; break
        if stop: break

    train_min = (time.time() - t_start) / 60
    model.save_pretrained(run_dir / "final")
    proc.save_pretrained(run_dir / "final")
    torch.save({"optimizer": optim.state_dict(), "step": step,
                "epoch": args.start_epoch + args.epochs},
               run_dir / "final" / "trainer_state.pt")   # 이어 학습용
    print(f"\n학습 종료 · {step} step · {train_min:.0f}분 · 건너뛴 샘플 {skipped}개")
    print(f"어댑터 저장: {run_dir/'final'}")

    # ── 4) 학습 후 평가 ──────────────────────────────────────────────────────
    print(f"\n{'─'*74}\n[3/3] 학습 후 평가")
    after = evaluate(model, proc, val_full, data_dir, args.image_size,
                     args.max_len, eval_batch, tag="after")
    report("학습 후", after, sub_mask, prev=before["acc"] if before else None)

    metrics.update({"after": {k: v for k, v in after.items() if k != "correct"},
                    "train_minutes": train_min, "steps": step, "skipped": skipped,
                    "loss_log": loss_log, "targets": "lm_only" if args.lm_only else "all"})
    if before:
        cb = np.array(before["correct"], bool); ca = np.array(after["correct"], bool)
        n10, n01 = int((ca & ~cb).sum()), int((cb & ~ca).sum()); n = n10 + n01
        p = math.erfc(math.sqrt(((abs(n10 - n01) - 1) ** 2 / n) / 2)) if n else 1.0
        metrics["mcnemar"] = {"fixed": n10, "broke": n01, "p": p}
        print(f"\nMcNemar: 학습이 고친 문항 {n10} · 망친 문항 {n01} · p={p:.4f} "
              f"{'유의' if p < .05 else '무의미'}")
        print("\n유형별 변화")
        for t in sorted(after["by_type"]):
            d = after["by_type"][t] - before["by_type"].get(t, 0)
            print(f"  {t:<4} {before['by_type'].get(t,0):.1%} → {after['by_type'][t]:.1%}  {d:+.1%}p")
    json.dump(metrics, open(run_dir / "metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n지표 저장: {run_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
