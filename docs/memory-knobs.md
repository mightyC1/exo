# Runtime-ручки форка: память Metal, дренаж кэшей, парковка, префилл
## v2 — полная инвентаризация по дереву (Август 2026)

Все дефолты вшиты в код — **обязательных env-флагов нет**. Штатный запуск:

```bash
EXO_GLM52_SPARSE_PREFILL=on \
EXO_MODELS_DIRS="/Volumes/Models:/Volumes/Models2" \
.venv/bin/exo --fast-sync
```

Длинноконтекстный штурм (150k+ на SSM-моделях):

```bash
EXO_PREFILL_STEP=1024 EXO_SSM_SNAPSHOT_EVERY=32 \
  EXO_GLM52_SPARSE_PREFILL=on EXO_MODELS_DIRS=... .venv/bin/exo --fast-sync
# бенч-таймаут >= 6000s; перед стартом kill -USR1 <раннер> (сброс парковки)
```

---

## 1. Контекст: инцидент «Resource limit (499000) exceeded»

Симптом: `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` на
длинных декодах / бенчах; крэш ранка посреди jaccl-коллектива вешал RDMA
соседей (QP→RTR errno 16 / EBUSY) вплоть до ребута всех нод.

Лимит — **счётчик живых Metal-буферов на процесс**, не байты. Четыре
механизма + промер потолка:

| # | Механизм | Модели | Скорость | Фикс |
|---|---|---|---|---|
| 1 | conv-state срезом без копии | Qwen3.5+ GDN | 1/GDN-слой/ток | апстрим #1077 (в пине) |
| 2 | неотсоединённые графы кэшей (mlx-lm #1332) | DSV4 | ~1/кэш/ток | дренаж, auto-гейт |
| 3 | парковка KVPrefixCache без cap по записям | все | ~1k буф/запрос навсегда | cap 32 + SIGUSR1 |
| 4 | ленивые цепочки `ArraysCache.advance()` (вне `state`) | linear-attn семейство (Qwen 48/ток, K3 66/ток) | 1/слой/ток | eval цепочек, безусловно |
| — | фолбэк 499000 консервативен ×8.4 | — | — | дефолт 2 000 000 в форке mlx |

Промерено (M3 Ultra / Tahoe, `~/rsrc-wall-test`): стена = **~2²² (4 194 304)
буфера на процесс**, отказ мягкий (nil-alloc → RuntimeError, не SIGABRT);
sysctl `iogpu.rsrc_limit` из ядра исчез; вставка дорожает ×5 к 4M.
Прод-лимит 2M = 48% стены.

Отдельный урок длинного контекста: скрэтч score-матриц префилла ∝
`chunk × kv` (эмпирика: peak−active = 60 GB на 110k при чанке 4096) —
рычаг `EXO_PREFILL_STEP`.

---

## 2. Ручки

### mlx (форк `mightyC1/mlx-jaccl-fix-small-recv`, ветка `mightyC1`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `MLX_RESOURCE_LIMIT` | — (в коде 2 000 000) | потолок Metal-буферов без пересборки; бьёт sysctl и фолбэк; мусор → фолбэк. `metal/device_info.cpp` |

Служебные mlx-distributed (ставит exo сам, руками не трогать):
`MLX_HOSTFILE, MLX_IBV_DEVICES, MLX_JACCL_COORDINATOR, MLX_RANK,
MLX_METAL_FAST_SYNCH, MLX_RING_VERBOSE`.

### Дренаж кэшей (`patches/opt_batch_gen.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_CACHE_DRAIN_EVERY` | `auto` | пер-степовый `mx.eval(cache.state)`: auto = GLM-5.2 IndexShare каждый шаг, DeepseekV4* по AUTO_EVERY, прочим 0. `N` — всем, `off` — никому |
| `EXO_CACHE_DRAIN_AUTO_EVERY` | `1` | интервал DSV4-ветки auto |
| `EXO_CACHE_DRAIN_ASYNC` | off | `1` — через async_eval, без стопа конвейера |
| `EXO_CACHE_DRAIN_WARN` | off | `1` — логировать проглоченные исключения дренажа |

Всегда включено, без ручки: **eval advance()-цепочек** (`_advance_chain_arrays`
в обоих пер-степовых async_eval) — механизм #4, стоимость ~0.

### Парковка префикс-кэшей (`cache.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_PREFIX_CACHE_MAX_ENTRIES` | `32` | LRU-cap числа записей (≈35k буферов max). `0` — легаси-безлимит |
| `EXO_MEMORY_THRESHOLD` | `0.85` (на ≥128GB) | байтовое вытеснение: LRU пока RAM% выше порога (psutil, distributed-max). Выше 0.9 не поднимать — упор в wired |

Ручной сброс без выгрузки: `kill -USR1 <pid раннера>` (только адресный kill).
Лог: `prefix-flush: dropped N parked cache entries`.

### Префилл — общие (`prefill_config.py`, `patches/ssm_snapshots.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_PREFILL_STEP` | `4096` (clamp 512–16384) | размер чанка префилла. Скрэтч ∝ step×kv: на 216k чанк 4096 даёт ~60+ GB транзиента, 1024 — ~15. Главный рычаг памяти длинного контекста; цена — чуть ниже tps |
| `EXO_SSM_SNAPSHOT_EVERY` | `1` | закладки SSM-состояний (ArraysCache: K3/KDA, Qwen/GDN; ~220MB/ранг/шт): `1` — каждый тик; `K` — каждый K-й; `off` — только пара отката. **Пара отката (total-1, total) берётся ВСЕГДА** — контракт «+2 rollback» префилла. `off` дополнительно пропускает парковку SSM-записей (без снапшота они нереюзабельны) |

### GLM-5.2 IndexShare префилл (`patches/glm52_prefill.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_GLM52_SPARSE_PREFILL` | `auto` | `on` — sparse-префилл всегда (×2.65 на 150k: 67→177 t/s); `auto` (дефолт) — sparse при kv ≥ MIN_KV; `off` — откат на dense |
| `EXO_GLM52_SPARSE_MIN_KV` | `12288` | кроссовер для auto: замерен 2026-09-01 (GLM-5.3 idxbf16, step 1024; dense/sparse пересекаются между 8k и 16k). При 0 auto = off (ворнинг в лог) |
| `EXO_GLM52_SPARSE_Q_CHUNK` | `256` | чанк запросов sparse-ветки |
| `EXO_GLM52_SPARSE_EVAL_BLOCKS` | off | пер-блочный eval sparse-пути (диагностика) |
| `EXO_GLM52_INDEXER_MODE` | `streaming` | streaming принят по A/B 157k: −5.9% времени, −11GB пика vs off; `reference` — диагностический; digit-accuracy 150k всё ещё в долгах |
| `EXO_GLM52_INDEXER_Q_CHUNK` | `256` | чанк запросов индексера |
| `EXO_GLM52_INDEXER_K_CHUNK` | `16384` | чанк ключей индексера (легаси-алиас `EXO_GLM52_INDEXER_CHUNK`) |
| `EXO_GLM52_INDEXER_EVAL_CHUNKS` | off | пер-чанковый eval индексера (диагностика) |
| `EXO_GLM52_SHARED_INDEX_CACHE` | `zero` | `compact` — компактный общий кэш индекса |
| `EXO_GLM52_CACHE_GROWTH` | — | политика роста кэша (см. `_parse_cache_growth`) |
| `EXO_GLM52_PREFILL_PROFILE` | off | профайлер префилла; `_SYNC` (деф. on), `_EVERY` (8), `_LAYER` — детализация |
| `EXO_SPARSE_ATTENTION_PATCHES` | `1` | **рубильник всего sparse-диспетча**; `0` — обойти патчи целиком |

### Kimi K3 (`vendor/kimi_k3.py`; семантика — в шапке файла)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_K3_NO_DECODE_COMPILE` | 0 | `1` — без `mx.compile(_decode_core)`. Бисекция авг-26: компайл чист, цена откл. ≈0 t/s |
| `EXO_K3_NO_CONV_KERNEL` | 0 | `1` — conv через апстримный ShortConv1d |
| `EXO_K3_GD_PREFILL_KERNEL` / `EXO_K3_GD_DECODE_KERNEL` | 0/0 | opt-in GD-Metal-кернелы. **Держать off**: GPU Address Fault в связке с RDMA (декод, 69 микрозапусков) |
| `EXO_K3_NO_GD_KERNEL` | 0 | форс-обход GD-кернелов поверх всего |
| `EXO_K3_NO_RES_KERNEL` | 0 | AttnRes-микс через ops вместо Metal-кернела |
| `EXO_K3_GD_SYNC_EVAL` | 0 | eval выходов кернела сразу после вызова (диагностика) |
| `EXO_K3_GD_COPY` | 0 | доп. копия выходов кернела в свежие буферы (диагностика) |

### Сторож/телеметрия (`patches/resource_guard.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_MLX_CACHE_DRAIN_EVERY` | `512` | `mx.clear_cache()` раз в N шагов декода; `0` — выкл |
| `EXO_MLX_MEM_LOG_EVERY` | `0` | `[RESGUARD]`-строка раз в N шагов |

### Инфраструктура / прочее (менять не приходится)

`EXO_HOME, EXO_RUNTIME_DIR, EXO_RESOURCES_DIR, EXO_DASHBOARD_DIR,
EXO_MODELS_DIRS, EXO_DEFAULT_MODELS_DIR, EXO_CUSTOM_MODEL_CARDS_DIR` — пути;
`EXO_FAST_SYNCH` (ставится флагом `--fast-sync`), `EXO_NO_BATCH` (флаг
`--no-batch`), `EXO_MAX_CONCURRENT_REQUESTS`, `EXO_OFFLINE`,
`EXO_BOOTSTRAP_PEERS`, `EXO_LIBP2P_NAMESPACE`, `EXO_MACMON_PATH`,
`EXO_ENABLE_IMAGE_MODELS`, `EXO_TRACING_ENABLED`.

---

## 3. Мониторинг и приёмка

`mtl-count.sh`: живые Metal-буферы (`-i N` системно; `--heap PID` пер-процессно).
`[PREFILL]`-телеметрия: `peak_bytes` < ~460 GB на худшем ранге; стабильный
`cache_bytes` (обвал = аллокатор каннибалит кэш, до jetsam недалеко).

Инварианты здорового кластера:
- декод любой модели: коридор ±50–500 буферов, без монотонного наклона;
- 50-вопросный прогон: потолок ≈ база + cap×(буф/запрос), плато, в логах
  `evicted LRU entry ... due to entry cap`;
- `kill -USR1` в покое роняет счётчик к базе;
- наклон = утечка: Δ/с ÷ t/s = буф/токен; число ≈ слоям называет виновника.

---

## 4. Долги в апстрим

1. **ml-explore/mlx**: sysctl исчез, фолбэк ×8.4 консервативен, стена 2²²,
   мягкий отказ, кривая вставки → PR/issue за `MLX_RESOURCE_LIMIT`
   (реализация готова в форке).
2. **ml-explore/mlx-lm**: `ArraysCache.advance()` строит ленивые цепочки вне
   `state` → 1 буфер/слой/токен у любого батч-потребителя. Воркэраунд —
   `_advance_chain_arrays`; апстрим-фикс: eager в advance либо включить
   `lengths`/`left_padding` в `state`.

Доказательная база: `~/rsrc-wall-test/probe-*.log`, mtl-count-логи прогонов.
