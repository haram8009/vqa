#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_submission.py — test 5,074장 추론 → submission.csv

파이프라인
----------
Qwen3-VL-4B (+ LoRA 어댑터) · a/b/c/d 로짓 비교 · 640px

안전장치
--------
1) **val 사전 점검** — test를 돌리기 전에 val 일부로 정확도를 먼저 잰다.
   어댑터가 안 붙었거나 토큰 id가 어긋난 상태로 20분을 태우는 사고를 막는다.
   기대치보다 크게 낮으면 경고하고 물어본다.
2) **중간 저장** — N배치마다 확률을 저장. 끊겨도 이어서 돌린다.
3) **제출 형식 검증** — sample_submission.csv와 id 집합·순서·행수를 대조하고,
   결측·비정상 문자·극단적 쏠림을 점검한 뒤에만 파일을 쓴다.

사용법
------
  python make_submission.py --val-check 150                       # 어댑터 없이 (zero-shot)
  python make_submission.py --adapter bakeoff_out/lora_qwen3vl/final
  python make_submission.py --adapter ... --image-size 640 --out submission_ep1.csv
  python make_submission.py --limit 50 --out /tmp/smoke.csv        # 파이프라인 점검
"""

from __future__ import annotations

import argparse, gc, inspect, json, math, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LETTERS = ["a", "b", "c", "d"]
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


def qtype(q):
    q = str(q)
    if "몇" in q:   return "개수"
    if "재질" in q: return "재질"
    if "색" in q:   return "색상"
    return "기타"


def build_mc_prompt(row):
    return (f"{row['question']}\n"
            f"(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n\n"
            "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")


def build_messages(row):
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user",   "content": [{"type": "image"},
                                           {"type": "text", "text": build_mc_prompt(row)}]}]


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


def load_model(dtype, use_4bit, adapter, merge=True):
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
            m = Cls.from_pretrained(MODEL_ID, **kw)
        except TypeError:
            kw["torch_dtype"] = kw.pop("dtype", dtype)
            m = Cls.from_pretrained(MODEL_ID, **kw)

    if adapter:
        from peft import PeftModel
        p = Path(adapter)
        if not p.exists():
            sys.exit(f"[중단] 어댑터 경로가 없습니다: {p}")
        m = PeftModel.from_pretrained(m, str(p))
        print(f"어댑터 로드: {p}")
        if merge and not use_4bit:
            # LoRA는 추론 때마다 모듈당 행렬곱 2번을 더한다. 베이스에 합쳐 없앤다.
            t = time.time(); m = m.merge_and_unload()
            print(f"LoRA 병합 완료 ({time.time()-t:.0f}s) — 추론 오버헤드 제거")
        elif merge and use_4bit:
            print("[!] 4bit에서는 LoRA 병합을 건너뜁니다 (양자화 가중치에 못 합침)")
    else:
        print("어댑터 없음 → zero-shot 추론")
    return m.eval()


def make_processor(image_size):
    from transformers import AutoProcessor
    px = image_size * image_size
    try:
        p = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=px, max_pixels=px,
                                          trust_remote_code=True)
    except TypeError:
        p = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    p.tokenizer.padding_side = "left"          # 로짓 스코어링에 필수
    return p


def get_letter_ids(tok, verbose=True):
    ids, rep = [], []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False) or tok.encode(" " + L,
                                                                   add_special_tokens=False)
        ids.append(enc[0]); rep.append((L, enc[0], repr(tok.decode([enc[0]]))))
    if verbose:
        print("  토큰 id:", ", ".join(f"{L}={i}({d})" for L, i, d in rep))
    assert len(set(ids)) == 4, f"[FAIL] 토큰 id 중복 {ids}"
    return ids


def predict(model, proc, df, data_dir, image_size, max_len, batch,
            letter_ids, desc="infer", cache=None, save_every=0):
    """확률 (N,4) 반환. cache 경로를 주면 중간 저장/재개."""
    import torch
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def load_image(row):
        img = Image.open(data_dir / row["path"]).convert("RGB")
        img.thumbnail((image_size * 2, image_size * 2), Image.BICUBIC)
        return img

    def prefetch(starts, ahead=3, workers=4):
        """다음 배치 이미지를 백그라운드에서 미리 디코딩 — CPU와 GPU를 겹친다."""
        def make(s):
            rows = [df.iloc[i] for i in range(s, min(s + batch, len(df)))]
            return rows, [load_image(r) for r in rows]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {s: ex.submit(make, s) for s in starts[:ahead]}
            for i, s in enumerate(starts):
                rows, imgs = futs.pop(s).result()
                nxt = i + ahead
                if nxt < len(starts):
                    futs[starts[nxt]] = ex.submit(make, starts[nxt])
                yield s, rows, imgs

    probs = np.full((len(df), 4), np.nan, dtype=np.float32)
    start = 0
    if cache and Path(cache).exists():
        try:
            saved = np.load(cache)
            if saved.shape == probs.shape:
                probs = saved
                done = ~np.isnan(probs[:, 0])
                start = int(done.sum())
                if start: print(f"  이어서 시작: {start}/{len(df)}장 완료됨")
        except Exception:
            pass
    if start >= len(df):
        return probs

    @torch.inference_mode()
    def score(rows, imgs=None):
        if imgs is None:
            imgs = [load_image(r) for r in rows]
        try:
            texts = [proc.apply_chat_template(build_messages(r), tokenize=False,
                                              add_generation_prompt=True) for r in rows]
            inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len).to(model.device)
            logits = model(**inputs).logits[:, -1, :].float()
            return logits[:, letter_ids].softmax(-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if len(rows) == 1: raise
            m = len(rows) // 2
            return np.concatenate([score(rows[:m], imgs[:m]),
                                   score(rows[m:], imgs[m:])], 0)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    starts = list(range(start, len(df), batch))
    t0 = time.time(); n_done = 0
    for k, (s, rows, imgs) in enumerate(
            tqdm(prefetch(starts), total=len(starts), desc=desc, unit="배치")):
        e = min(s + batch, len(df))
        probs[s:e] = score(rows, imgs)
        n_done += e - s
        if cache and save_every and k % save_every == 0:
            np.save(cache, probs)
    if cache:
        np.save(cache, probs)
    el = time.time() - t0
    if n_done:
        print(f"  {el/60:.1f}분 · {el/n_done:.3f}s/장")
    return probs


def main():
    ap = argparse.ArgumentParser(description="test 추론 → submission.csv")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--adapter", default="", help="LoRA 어댑터 경로 (없으면 zero-shot)")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=0, help="0=VRAM 보고 자동")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--4bit", dest="use_4bit", action="store_true",
                    help="기본은 bf16 (추론만 하므로 VRAM 여유). OOM 시 사용")
    ap.add_argument("--val-check", type=int, default=150,
                    help="test 전에 val 몇 장으로 점검할지. 0=건너뜀")
    ap.add_argument("--expect-acc", type=float, default=0.75,
                    help="val 점검이 이 값 미만이면 경고하고 중단 여부를 묻는다")
    ap.add_argument("--limit", type=int, default=0, help="test 앞 N장만 (스모크용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--no-merge", action="store_true",
                    help="LoRA를 베이스에 병합하지 않음 (기본은 병합해서 더 빠르게)")
    ap.add_argument("--workers", type=int, default=4, help="이미지 디코딩 스레드 수")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 진행")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()
    for f in ("test.csv", "sample_submission.csv", "test"):
        if not (data_dir / f).exists():
            sys.exit(f"[중단] {data_dir/f} 없음. --data-dir 확인")

    test_df = pd.read_csv(data_dir / "test.csv")
    test_df["qtype"] = test_df["question"].map(qtype)
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    # 제출 형식 사전 대조
    assert list(sample.columns) == ["id", "answer"], f"sample 컬럼이 다릅니다: {list(sample.columns)}"
    if not (sample["id"].values == test_df["id"].values).all():
        print("[!] sample_submission과 test.csv의 id 순서가 다릅니다 → sample 순서에 맞춥니다")
        test_df = test_df.set_index("id").loc[sample["id"]].reset_index()
    if args.limit:
        test_df = test_df.head(args.limit).reset_index(drop=True)

    print("=" * 74)
    print(f"모델   {MODEL_ID}" + (f" + {args.adapter}" if args.adapter else " (zero-shot)"))
    print(f"데이터 {data_dir}")
    print(f"test   {len(test_df)}장 · 해상도 {args.image_size} · "
          f"{'4bit NF4' if args.use_4bit else 'bf16'}")
    print(f"출력   {out_path}")
    print("=" * 74)

    import torch
    if not torch.cuda.is_available(): sys.exit("[중단] CUDA 없음")
    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    # 추론 전용이라 학습보다 메모리가 훨씬 여유롭다 → 배치를 크게 잡는다.
    # OOM이 나도 score()가 재귀적으로 반씩 쪼개므로 안전하다.
    batch = args.batch or (16 if vram >= 22 else (8 if vram >= 14 else 2))
    if args.image_size >= 768: batch = max(1, batch // 2)
    print(f"\nGPU {torch.cuda.get_device_name(0)} ({vram:.1f}GB) · "
          f"dtype {str(dtype).split('.')[-1]} · batch {batch}")

    proc  = make_processor(args.image_size)
    model = load_model(dtype, args.use_4bit, args.adapter, merge=not args.no_merge)
    print(f"VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    letter_ids = get_letter_ids(proc.tokenizer)

    # ── 1) val 사전 점검 ─────────────────────────────────────────────────────
    if args.val_check:
        print(f"\n{'─'*74}\n[1/2] val {args.val_check}장 사전 점검 "
              f"(어댑터·토큰 id·해상도가 제대로 붙었는지 확인)")
        tr = pd.read_csv(data_dir / "train.csv")
        tr["qtype"] = tr["question"].map(qtype)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(tr))
        val = tr.iloc[perm[:int(len(tr) * args.val_frac)]].reset_index(drop=True)
        val = val.head(args.val_check)
        vp = predict(model, proc, val, data_dir, args.image_size, args.max_len,
                     batch, letter_ids, desc="val점검")
        pred = np.array([LETTERS[i] for i in vp.argmax(1)])
        gold = val["answer"].str.strip().str.lower().values
        acc = float((pred == gold).mean())
        print(f"  val 정확도 {acc:.1%} (n={len(val)})")
        print("  예측분포 " + " ".join(f"{L}:{(pred==L).mean():.0%}" for L in LETTERS))
        if acc < args.expect_acc:
            print(f"\n  [!] 기대치 {args.expect_acc:.0%}보다 낮습니다. 점검하세요:")
            print("      · 어댑터 경로가 맞는가 (--adapter)")
            print("      · 해상도가 학습 때와 같은가 (--image-size)")
            print("      · 토큰 id 4개가 서로 다른가 (위 출력)")
            print("      · 예측이 한 글자로 쏠려 있지 않은가")
            if not args.yes:
                if input("\n  그래도 test 추론을 진행할까요? [y/N] ").strip().lower() != "y":
                    sys.exit("중단했습니다.")
        else:
            print("  → 정상. test 추론으로 넘어갑니다.")

    # ── 2) test 추론 ─────────────────────────────────────────────────────────
    print(f"\n{'─'*74}\n[2/2] test {len(test_df)}장 추론")
    cache = None if args.no_cache else str(out_path.with_suffix(".probs.npy"))
    probs = predict(model, proc, test_df, data_dir, args.image_size, args.max_len,
                    batch, letter_ids, desc="test", cache=cache, save_every=20)

    # ── 3) 제출 파일 검증 후 저장 ────────────────────────────────────────────
    nan_rows = int(np.isnan(probs[:, 0]).sum())
    if nan_rows:
        sys.exit(f"[중단] 추론이 안 된 행이 {nan_rows}개 있습니다. 다시 실행하면 이어서 돌립니다.")

    pred = np.array([LETTERS[i] for i in probs.argmax(1)])
    sub = pd.DataFrame({"id": test_df["id"].values, "answer": pred})

    print("\n[제출 형식 검증]")
    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
        ok = ok and cond

    chk(len(sub) == len(sample) or args.limit, f"행 수 {len(sub)} (sample {len(sample)})")
    chk(list(sub.columns) == ["id", "answer"], f"컬럼 {list(sub.columns)}")
    chk(sub["answer"].isin(LETTERS).all(), "answer가 전부 a/b/c/d")
    chk(sub["id"].notna().all() and sub["answer"].notna().all(), "결측 없음")
    chk(sub["id"].duplicated().sum() == 0, "id 중복 없음")
    if not args.limit:
        chk((sub["id"].values == sample["id"].values).all(), "id 순서가 sample과 동일")
    dist = {L: float((pred == L).mean()) for L in LETTERS}
    skew = max(abs(v - .25) for v in dist.values())
    chk(skew < .15, f"예측 분포 쏠림 {skew:.3f} (< 0.15) — " +
        " ".join(f"{L}:{v:.0%}" for L, v in dist.items()))

    if not ok:
        sys.exit("\n[중단] 검증 실패. 위 FAIL 항목을 확인하세요.")

    sub.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
    print(sub.head(3).to_string(index=False))

    meta = {"model": MODEL_ID, "adapter": args.adapter or None,
            "image_size": args.image_size, "dtype": str(dtype).split(".")[-1],
            "use_4bit": args.use_4bit, "n_test": len(sub), "pred_dist": dist,
            "qtype_dist": {k: float(v) for k, v in
                           test_df.qtype.value_counts(normalize=True).items()}}
    json.dump(meta, open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"메타: {out_path.with_suffix('.meta.json')}")
    if cache:
        print(f"확률: {cache}  (나중에 앙상블·분석용)")


if __name__ == "__main__":
    main()
