# 실험 로그 04 — remote-code VLM 로딩 트러블슈팅

- 날짜: 2026-08-28
- 환경: RTX 5060 Ti 16GB (sm_120) · torch 2.7.0+cu128 · transformers (git 최신) · Linux 컨테이너
- 대상: `kakaocorp/kanana-1.5-v-3b-instruct` (베이크오프 3모델 중 유일하게 로딩이 깨진 모델)
- 상태: 해결. 대응책은 `kanana_bakeoff.py`, `step1_2_bakeoff*.ipynb`, `resolution_experiment.py`, `lora_train.py`에 이식됨

Qwen 계열 두 모델은 한 번에 로드됐다. kanana만 **세 번 연속으로** 다른 이유로 실패했다. 셋 다 remote-code 모델(`trust_remote_code=True`)에서 일반적으로 만날 수 있는 문제라 순서대로 기록한다.

---

## 실패 1 — Auto 클래스를 추측하면 안 된다

```
ValueError: Unrecognized configuration class KananaVConfig
for this kind of AutoModel: AutoModelForImageTextToText.
Model type should be one of AriaConfig, AyaVisionConfig, ...
```

### 원인

kanana의 `config.json`은 `auto_map`에 **`AutoModelForVision2Seq` 하나만** 등록한다.

```json
"auto_map": {
  "AutoConfig": "configuration.KananaVConfig",
  "AutoModelForVision2Seq": "modeling.KananaVForConditionalGeneration",
  "AutoImageProcessor": "processing_image.KananaVImageProcessor",
  "AutoProcessor": "processing.KananaVProcessor"
}
```

내 코드는 최신 정식 클래스인 `AutoModelForImageTextToText`를 우선 시도했다. transformers는 `auto_map`에서 그 이름을 못 찾자 **내장 모델 목록**에서 `KananaVConfig`를 찾다가 실패했다. Qwen은 내장 모델이라 이 경로를 타지 않았다.

### 왜 "auto_map이 지정한 Auto 클래스를 쓴다"로는 부족한가

**transformers 5.x에는 `AutoModelForVision2Seq`가 아예 없다** (`AutoModelForImageTextToText`로 통합). 검증:

```
transformers 5.16.1
AutoModelForVision2Seq 존재: False
AutoModelForImageTextToText 존재: True
```

버전에 따라 없거나 별칭이라, **Auto 클래스 이름만으로는 못 맞춘다.**

### 해결 — 동적 모듈에서 실제 클래스를 직접 로드

```python
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

cfg  = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
amap = getattr(cfg, "auto_map", None) or {}
ref  = next((amap[k] for k in ("AutoModelForVision2Seq",
                               "AutoModelForImageTextToText",
                               "AutoModelForCausalLM") if k in amap), None)
Cls  = get_class_from_dynamic_module(ref, model_id)   # Auto 등록 절차를 통째로 우회
```

Auto 플럼빙을 건너뛰므로 **버전 무관**하다. `auto_map`이 없는 표준 모델(Qwen)은 기존대로 `AutoModelForImageTextToText`로 간다.

오프라인 재현 테스트에서 분기가 정확히 갈리는 것을 확인했다 — kanana는 동적 로드 경로, Qwen은 표준 Auto 경로.

---

## 실패 2 — remote code의 의존성은 미리 알 수 없다

```
Encountered exception while importing timm: No module named 'timm'
ImportError: This modeling file requires the following packages
that were not found in your environment: timm. Run `pip install timm`
```

kanana의 vision encoder가 timm 기반이다(`from timm.layers import LayerNorm, LayerNorm2d, resample_abs_pos_embed`, `from timm.models.regnet import RegStage`). 설치하면 다음은 `einops`가 나온다.

### 해결 — 에러 메시지에서 패키지명 파싱 → 자동 설치 → 재시도

두 가지 메시지 형태를 모두 처리한다.

```python
_DEP_RE = re.compile(r"not found in your environment:\s*([^\.\n]+)")
_MOD_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
```

**재시도 전에 `sys.modules` 정리가 필수다.** 반쯤 import된 remote code 모듈이 남으면 재시도가 계속 깨진다.

```python
for k in [k for k in list(sys.modules) if k.startswith("transformers_modules")]:
    del sys.modules[k]
```

`flash_attn`은 자동 설치 대상에서 **명시적으로 제외**한다(실패 3 참고).

---

## 실패 3 — `flash_attention_2`가 try/except를 뚫는다

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used
due to the following error: the package flash_attn seems to be not installed.
```

kanana의 `modeling.py`에는 **분명히 폴백이 있다.**

```python
try:
    self.vision_model = CustomQwen2VLVE._from_config(
        config.vision_config, attn_implementation="flash_attention_2")
except Exception as e:
    logger.error(e)
    logger.info("Failed to load Vision Encoder with flash_attention_2. Try without it ...")
    self.vision_model = CustomQwen2VLVE._from_config(config.vision_config)
```

`ImportError`는 `Exception`의 하위 클래스이므로 잡혀야 한다. 그런데 밖으로 나온다.

### 원인 — config 오염

1차 호출이 실패하면서 transformers가 **`config.vision_config._attn_implementation`을 `'flash_attention_2'`로 이미 써버린다.** 폴백 호출은 그 오염된 config를 그대로 쓰므로 같은 flash를 다시 요구하고, 이번엔 잡아주는 곳이 없다.

작은 Llama 모델로 재현했다.

```
1차 실패: ImportError  · config 오염 → '_attn_implementation' = 'flash_attention_2'
폴백 실패: ImportError                ← 이게 밖으로 나오는 에러
```

**config를 미리 청소해도 소용없다** — 1차 호출이 다시 오염시킨다. (이 방법을 먼저 시도했다가 실패했다.)

### 해결 — transformers 진입점을 일시 몽키패치

flash 요청 자체가 도달하지 못하게 막는다.

```python
@contextmanager
def no_flash_attention(impl="sdpa"):
    from transformers.modeling_utils import PreTrainedModel as PM
    for name in ("_from_config",                            # 모든 버전
                 "_check_and_adjust_attn_implementation",   # 4.5x+
                 "_autoset_attn_implementation"):           # 구버전
        # flash_attention_2 문자열을 impl로 치환하는 래퍼로 교체
        # (classmethod/staticmethod/일반 메서드 구분해서 원본 디스크립터 보존)
    try:
        yield
    finally:
        ...  # 블록을 벗어나면 원복
```

검증 결과:

```
impl=sdpa : 1차 sdpa / 폴백 sdpa
impl=eager: 1차 eager / 폴백 eager
컨텍스트 종료 후 → 다시 ImportError (원복 확인)
```

### `flash_attn` 설치는 해결책이 아니다

- FlashAttention-2는 **Ampere(sm80) 이상 전용**. T4(sm75)에서는 애초에 못 쓴다
- 빌드에 10분 이상 걸린다
- 이 대회 환경에서 얻는 게 없다

자동 설치 대상에서 제외하고, 패치를 뚫고 나오면 **같은 에러로 무한 재시도하지 않도록 즉시 중단**하고 `--attn eager` 안내를 띄운다.

---

## 그 밖에 겪은 것

### `padding_side="left"` 필수

로짓 스코어링은 `logits[:, -1, :]`를 읽는다. 오른쪽 패딩이면 그 자리가 패딩 토큰이라 정확도가 조용히 무너진다.

```python
processor.tokenizer.padding_side = "left"
```

### a/b/c/d 토큰 id를 모델마다 확인

토크나이저에 따라 `"a"`와 `" a"`가 갈린다. 잘못된 id를 잡으면 **정확도가 조용히 25% 근처에 고정된다.** 모델을 바꿀 때마다 assert.

```python
assert len(set(letter_ids)) == 4, f"토큰 id 중복 {letter_ids}"
```

### sm80 미만은 bfloat16 미지원

T4(Colab)는 bf16을 네이티브로 못 쓴다. compute capability로 자동 선택.

```python
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
```

로짓은 softmax 전에 `.float()`로 승격해 fp16 오버플로를 막는다.

### matplotlib 한글 폰트

`fm.fontManager.ttflist`는 **캐시 기반**이라 오래되면 시스템 폰트를 통째로 놓친다. Windows에서 맑은고딕이 있는데도 □가 나오는 전형적인 원인.

이름으로 찾지 말고 **폰트 파일 경로를 직접 훑고, 글리프 유무를 검증한 뒤 등록**한다.

```python
from matplotlib import ft2font
def has_hangul(path):
    return ft2font.FT2Font(path).get_char_index(ord("한")) != 0
# 알려진 경로 우선 → 실패 시 findSystemFonts() 전수 조사 → addfont()
```

전수 조사에는 우선순위가 필요하다. 그냥 "글리프가 있는 첫 폰트"를 고르면 `Unifont Sample` 같은 비트맵 폴백 폰트를 집는다(실제로 겪음). 맑은고딕·나눔·Noto CJK 계열을 우선하고 Unifont·Symbol류는 후순위로 밀었다. 끝내 못 찾으면 **그래프 라벨을 영어로 자동 전환**한다 — □보다는 낫다.

### `Skipping import of cpp extensions due to incompatible torch version`

무시해도 되는 경고. 별개 패키지가 최적화 확장을 건너뛴다는 안내일 뿐 동작에 영향 없다.

---

## 정리 — remote-code 모델을 붙일 때의 순서

1. `config.auto_map`을 읽고 **동적 모듈에서 클래스를 직접 로드**한다. Auto 클래스 이름을 추측하지 않는다
2. 첫 로드는 **의존성 자동 설치 + 재시도** 루프로 감싼다. 재시도 전 `transformers_modules*`를 `sys.modules`에서 제거
3. `no_flash_attention` 컨텍스트 안에서 로드한다. **`flash_attn`은 설치하지 않는다**
4. `padding_side="left"` 설정, 토큰 id 4개 중복 여부 assert
5. **8장짜리 스모크 테스트를 먼저 돌린다** — 실패해도 10분이 아니라 1분 만에 안다

5번이 특히 도움이 됐다. kanana는 세 번 실패했는데, 매번 300장 전체를 돌리는 대신 8장으로 끊었으면 훨씬 빨리 끝났을 것이다. `kanana_bakeoff.py`에 `--smoke`와 `--dry-run`을 넣은 이유다.

---

## 부록 · 재현

```bash
python kanana_bakeoff.py --dry-run   # 데이터·경로·기준선만 (10초)
python kanana_bakeoff.py --smoke     # 8장, 로딩·토큰 id 확인 (1분)
python kanana_bakeoff.py             # 본 측정
python kanana_bakeoff.py --attn eager   # flash 우회가 안 될 때
```

대응 코드 위치: `kanana_bakeoff.py`의 `no_flash_attention` / `_resolve_kanana_class` / `_missing_packages` / `_clear_remote_modules`.
