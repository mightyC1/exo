# GLM-5.2/5.3 MTP speculative decoding — recon (Фаза 0)

Статус: **ГЕЙТ 0-A закрыт**, go на side-car экстракцию → Фаза 1 (shadow).
База: ветка `mightyC1` @ `0b5fe626` (site-defaults W-серии). Пины:
mlx-lm `rltakashige @ 6a3df6cd`, mlx `mightyC1/mlx-jaccl-fix-small-recv @ cc3f3e60`.
ТЗ: glm52-mtp-tz-v1. Все file:line ниже проверены глазами по пину/дереву.

## 1. Аудит весов (§3.1) — итог

`sanitize` пина дропает `layers.N ≥ num_hidden_layers` (ds32:513–521) →
в обоих idxbf16-конвертах **0** MTP-тензоров (скан index.json). Источники:

| | GLM-5.2 (`zai-org/GLM-5.2`) | GLM-5.3 (`zai-org/GLM-5.3`) |
|---|---|---|
| формат слоя 78 | **bf16** (без scale_inv), 791 тензор / 5 шардов | **fp8 e4m3 + block-128 scale_inv**, 1569 (791+778 скейлов) / 3 шарда |
| bf16-исключения вендора | всё | eh_proj, indexer.weights_proj, k_norm, нормы, mlp.gate (`modules_to_not_convert`) |

Состав слоя 78 = полный DecoderLayer семейства + MTP-обвес:
MLA-attention (q_a/q_b/kv_a/kv_b/o + нормы) + **собственный полный индексер**
(wq_b [4096,2048] / wk [128,6144] / weights_proj [32,6144] / k_norm) +
MoE (256 routed + shared_experts + gate c e_score_correction_bias) +
`enorm/hnorm/eh_proj [6144,12288]` + `shared_head.norm`. `shared_head.head`
отсутствует → lm_head **tied** с главной моделью (как и embed) — схема ТЗ §4
подтверждена побайтно по именам. Шейпы согласуются с конфигом
(H=64, nope=192, rope=64, v=256, kv_lora=512, q_lora=2048, hidden=6144).

Конфиги: `num_nextn_predict_layers=1`, `index_share_for_mtp_iteration=true`,
`indexer_types` длиной **78** (слой 78 разметкой не покрыт). idxbf16 quant:
`{group_size:64, bits:8, mode:affine}`; per-module False-флаги индексера есть
только в 5.2-конфиге — косметика двух прогонов конверта, рантайм-поведение
presence-driven (наличие `<name>.scales`), side-load делаем так же.

**Развилка индексера закрыта:** у MTP-слоя свой полный индексер → блок
самодостаточен, top-k из главного прохода не пробрасываем;
`index_share_for_mtp_iteration` в v1 игнорируем (семантика verify-итерации
serving-движков). Vendor-тень `indexers_proj` в modules_to_not_convert —
опечатка, реальный тензор `indexer.weights_proj`.

## 2. Side-car политика (tools/glm52_mtp_extract_sidecar.py)

Та же политика, что у главного idxbf16-конверта:
1. fp8→bf16 — **дословная** математика dequant из sanitize пина
   (ds32:524–539, block-128, pad+reshape+scale_inv); 5.2 — passthrough.
2. Трансформы sanitize для слоя 78: expert-stack `experts.{e}` →
   `switch_mlp` (ds32:553–563); split `kv_b_proj` → `embed_q/unembed_out`
   (ds32:565–599).
3. int8 g64 affine: q_a/q_b/kv_a/o, embed_q/unembed_out, switch_mlp×3,
   shared_experts×3 (12 модулей).
4. bf16 навсегда: весь индексер (урок idxbf16), eh_proj и mlp.gate.weight
   (вслед за вендором; routing-чувствительно), все нормы;
   `e_score_correction_bias` — f32 (cast_predicate пина, ds32:666).
5. Выход: `mtp.safetensors` (51 тензор) + `mtp.manifest.json` (sha256,
   политика) рядом с моделью; `EXO_GLM52_MTP_WEIGHTS=auto` ищет там.

Parity-валидация (`--self-test`, реальные операторы, CPU-mlx + пин):
dequant / switch_mlp / embed_q+unembed_out **байт-в-байт** против живого
`Model.sanitize`; int8 roundtrip sane. E2E на синтетике — оба режима
источника (bf16 и fp8) → 51/12/15, зелёный.

## 3. Доноры (§3.2) — вердикт: standalone `patches/glm52_mtp.py`

`alexcheema/mtp-speculative-decoding` (DeepSeek-V3, +1660):
лифтим композицию блока (enorm/hnorm/eh_proj + decoder-block + shared
embed/lm_head/final-norm), greedy-accept, откат через `trim_prompt_cache`.
Дефекты (не тащить): кормит **post-norm** hidden (`model()` возвращает
`norm(h)`, ds32:494 → double-norm); MTP-кэш без префилла и с offset 0
(RoPE-позиции врут на prompt_len); standalone-генератор мимо continuous
batching; dense-MLP-упрощение (наш слой 78 — MoE); веса патчем sanitize.

`david/speculative-mtp` (Qwen3.5, +5144): лифтим идеи — захват **pre-norm**
hidden обёрткой `inner.norm`; честный rejection sampling
(u < min(1, p/q), ресемпл из normalize(max(0,p−q)), bonus из p);
каркас batch-generator: гейт B==1 + прозрачный fallback + буфер принятых
токенов с выдачей по одному на шаг; стоп-обрезка внутри окна; chained-γ
(Фаза 4). Не тащим: GDN/кернелы, откат мутацией `offset -=` мимо `trim()`,
нестандартный `ratio**alpha`.

## 4. Точки интеграции (§3.3), проверено по дереву

- Декод-шаг: `mlx_lm.generate.GenerationBatch._step` (generate.py:1344),
  **уже перехвачен** `patches/opt_batch_gen._patched_step` (:180, установка
  :270 из bootstrap `apply_mlx_patches()`). Хук MTP — обёртка поверх
  установленного `_step` (сохранённый — fallback). Гейт: env on/shadow ∧
  family glm_moe_dsa ∧ `len(self.uids)==1` ∧ mtp.safetensors валиден;
  `buf.needs_topk` (logprobs) → warn-once + fallback (v1).
- Пайплайнинг: `_step` выдаёт токен предыдущей итерации
  (`_current=_next`, async_eval следующего) — MTP-цикл сохраняет инвариант:
  y = pending `_next_tokens`, verify L=1+k, новый `_next_tokens` =
  bonus/corrected, принятые — в буфер.
- Pre-norm hidden: обёртка `model.model.norm` (Дэвид-паттерн). TP `shard()`
  (ds32:603–659) шардит только внутренности слоёв; embed/norm/lm_head/
  indexer реплицированы → h и логиты реплицированы (факт 4 ТЗ ✓).
- Кэши: `CacheList.trim(n)` трогает оба слота (cache.py:824–827);
  shared-слои двигают cache[1] placeholder-апдейтом каждый forward
  (glm52_indexshare.py:436–448) → офсеты слот-0/слот-1 равны на всех 78
  слоях, откат единообразен. Смок в контейнере (реальные операторы):
  fwd6 vs fwd6→trim(2)→redo — **байт-в-байт**, оффсеты (6,6)→(4,4).
- Непатченные attention-инстансы (наш MTP-блок) уходят в
  `_exo_original_call` (glm52_indexshare.py:318) — IndexShare-контекст
  главной модели не задевается.
- W2-fingerprint собирает весь `EXO_GLM52_*` (prefill_config.py:42–51) →
  `EXO_GLM52_MTP*` в кросс-ранговом хэше автоматически (§9 ТЗ ✓).
- `mx.random.seed(seed)` + per-request sampler — batch_generate.py:185–194.

## 5. Дизайн-заметки для Фазы 1 (сверх ТЗ)

- D1. Блок = `DeepseekV32DecoderLayer(config, layer_idx=78)` пина: MoE-ветка
  выбирается автоматически (ds32:398–413), индексер в комплекте. Реплика на
  всех рангах, без шардинга (≈10.6GB int8/ранг).
- D2. MTP-кэш = свой `CacheList(KVCache, KVCache)`; lockstep trim с
  главными. v1 без префилла MTP-кэша по промпту → RoPE-позиции драфтера
  относительные. Если shadow-accept < гейта — первая мера: MTP-префилл
  по чанкам главного префилла (стоимость ~1/78 ≈ +1.3%) либо кастомный
  OffsetKVCache (виртуальный базовый offset). Решение по данным shadow.
- D3. M2: p и q для rejection sampling обязаны проходить **одинаковые**
  трансформации сэмплера (temp/top_p/min_p/top_k), иначе смещение.
- D4. Финиш/стоп внутри окна: перед `extract_cache` кэш триммится до
  фактически выданных токенов (иначе prefix-cache получит хвост-сироту).
  Тест в G2-A набор.
- D5. `buf.needs_topk` → fallback (v1); topk-конвейер под окно — не сейчас.
- D6. Загрузка side-car presence-driven: модуль квантован ⇔ есть
  `<name>.scales` в mtp.safetensors; конфиг-флаги не читаем.
- D7. Донорский `mtp_cache` без trim на reject у Алекса — при k=1 запись
  позиции валидна (считана от принятого (h,t)), но наша хореография всё
  равно lockstep-trim по ТЗ §5.5 — единый инвариант офсетов проще
  ассертить (`EXO_GLM52_MTP_VALIDATE=1`).

## 6. Дальше

1. Прогнать экстрактор по обоим источникам, sha256/счётчики в этот док
   (append) — закрытие 0-A артефактом.
2. Фаза 1: `patches/glm52_mtp.py` — блок + side-load + shadow-режим,
   телеметрия `[MTP_SHADOW]`; сутки на OpenCode-трафике; ГЕЙТ 1-A ≥60%.

---

## Итоги фаз 1–2 (2026-09-01) — что измерено, что решено

### Факты, закрывающие вопросы recon
- **Порядок eh_proj**: vLLM `deepseek_mtp.py` (обслуживает `glm_moe_dsa`):
  `eh_proj(cat([enorm(embeds), hnorm(hidden)]))` → дефолт `eh` верен; у донора
  `alexcheema` порядок перевёрнут (ещё один дефект донора). Вход MTP — **pre-norm** h
  (их же комментарий у compute_logits). vLLM форсит fp32-роутинг MoE для glm_moe_dsa —
  наш блок использует MoEGate пина, как и все 78 главных слоёв.
- **Декод-кэши — `BatchKVCache`** (граduация prompt→generation оборачивает plain
  KVCache), trim = `_idx/offset -= n`, offset — массив. Уточняет §2.1.
- **Ординал-1 аномалия**: первый запрос после загрузки численно ≠ последующим
  (h0sum 117.6 vs −82.4 при одинаковом префилле). Warm-слоты воспроизводимы.
  Протокол замеров: `--warmup 1`, сравниваем только warm-слоты.
- **fast-sync и нож**: под `--fast-sync` (штатный режим кластера) траектории при
  temp=0 плавают на ножевых токенах (маржи 0.125–0.375 = шаг bf16-сетки);
  VALIDATE=1 маскировал это пер-цикловым барьером (цена барьера ~1ms). Off под
  fast-sync подвержен так же. На живом корпусе (код+доки) ранние маржи жирные,
  ординал-1 и нож не воспроизводятся.
- **G2-A′**: побайтность on≡off кросс-L (L=2 verify против L=1 декода) на bf16
  недостижима и ни vLLM, ни SGLang её не обещают; гарантия — консистентность
  относительно собственного verify (enforced конструкцией + VALIDATE). Арбитр
  качества — G2-B digit-150k on vs off.

### Shadow (фаза 1)
Accept при temp=1.0 на бенч-генерации 0.83–0.93 (win256 до 0.94); paттерн
«низкий старт (пустой суффиксный MTP-кэш) → рост». Налог тени −1.7%.

### Три бага маршрутизации verify L=2 (все исправлены, 0006/0007/0011)
1. `cache_requires_dense_prefill` по наличию `left_padding` → dense по всему kv
   каждый цикл (0.69 t/s на 49k) + сдвиг accept (dense-h против sparse-мира).
2. `sparse_min_kv` гнал L=2 в dense-префилл-ветку при kv<12288 → micro-L
   (`_MICRO_DECODE_L=4`) всегда sparse при наличии topk.
3. tiled/streaming индексер для L=2 → монолит (экономии не дал, оставлен как
   корректный путь).

### Боевой M1 — матрица (warm, temp=0, корпус код+доки, VALIDATE=0)
| pp | off | on | прирост | accept | цикл |
|---|---|---|---|---|---|
| 4096 | 19.05 | 20.9–22.0 | +10–16% | 0.61–0.69 | 1.45–1.49 шага |
| 49152 | 18.55 | 22.8–23.1 | +23–24% | 0.82–0.85 | 1.48 |
| 147456 | 16.99 | 20.3–20.8 | +20–22% | 0.77–0.85 | 1.48 |

Профиль цикла (49k): build 8.2 / resolve 68.6 / post 2.3 ms — синхронный такт
(host резолвит m до всего остального; пин прячет ~14ms однотактным конвейером).
Accept на глубине **растёт** (продолжение документа предсказуемее свободного
ответа) → D2-прогрев MTP-кэша не нужен.

### M1.5 (чейнинг) — отрицательный результат, откачен
`mx.async_eval` допускает одну оценку в полёте (второй вызов блокируется до
завершения первой — измерено); slice-присваивание mx.array мутирует объект
in-place, выброшенный build-ahead граф остаётся в lineage кэша и считается.
Следствие: для k=1 синхронный такт — потолок структуры; цель ≥28 t/s на kv≥50k
берётся уменьшением работы на цикл — **k=2 цепные драфты (фаза 4)**, оценка
eff≈2.3–2.5 при accept₂≈0.6 → ~29–31 на 49k.

### Открытое
- G2-B digit-150k on/off (сиды 0/4/5) — арбитр качества.
- M2 (honest rejection sampling) — ускорение OpenCode (temp=1.0), сейчас его
  трафик под `on` идёт shadow-путём.
- Ординал-1 численность первого исполнения — тикет базового стека.
