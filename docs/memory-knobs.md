# Runtime-ручки форка: память Metal, дренаж кэшей, парковка
## Итог инцидента «Resource limit (499000) exceeded» (Август 2026)

Все дефолты вшиты в код — **обязательных env-флагов нет**. Штатный запуск:

```bash
EXO_GLM52_SPARSE_PREFILL=on EXO_GLM52_SPARSE_Q_CHUNK=256 \
EXO_MODELS_DIRS="/Volumes/Models:/Volumes/Models2" \
.venv/bin/exo --fast-sync
```

---

## 1. Контекст: что чинилось

Симптом: `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` на
длинных декодах / бенчах; на меше крэш ранка посреди jaccl-коллектива вешал
RDMA соседей (QP→RTR errno 16 / EBUSY) вплоть до ребута всех нод.

Лимит — **счётчик живых Metal-буферов на процесс**, не байты
(`num_resources_` в аллокаторе mlx). Найдено и закрыто четыре независимых
механизма + промерен реальный потолок железа:

| # | Механизм | Модели | Скорость утечки | Фикс |
|---|---|---|---|---|
| 1 | conv-state срезом без копии (вью держит родителя) | Qwen3.5+ GDN | 1/GDN-слой/токен | апстрим #1077 (в пине есть) |
| 2 | неотсоединённые графы pooling/indexer-кэшей (mlx-lm #1332) | DSV4 | ~1/кэш/токен | пер-степовый дренаж, auto-гейт |
| 3 | парковка KVPrefixCache без вытеснения по числу записей | все | ~1k буферов/запрос, навсегда | cap 32 + SIGUSR1-флаш |
| 4 | ленивые цепочки `ArraysCache.advance()` поверх `lengths`/`left_padding` (не входят в `state`, батч-путь EXO) | всё linear-attention семейство (Qwen GDN 48/ток, K3 KDA 66/ток) | 1/linear-слой/токен | eval цепочек в пер-степовом async_eval, безусловно |
| — | фолбэк лимита 499000 консервативен ×8.4 | — | — | дефолт 2 000 000 в форке mlx |

Промерено на M3 Ultra / Tahoe (`~/rsrc-wall-test`, авг 2026): реальная стена
= **~2²² (4 194 304) буфера на процесс**, отказ мягкий (`nil`-alloc →
RuntimeError, не SIGABRT); sysctl `iogpu.rsrc_limit` из ядра Tahoe исчез;
стоимость вставки в residency-set растёт ×5 к 4M. Прод-лимит 2M = 48% стены.

---

## 2. Ручки

### mlx (форк `mightyC1/mlx-jaccl-fix-small-recv`, ветка `mightyC1`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `MLX_RESOURCE_LIMIT` | — (в коде 2 000 000) | переопределяет потолок Metal-буферов без пересборки; бьёт и sysctl, и фолбэк; мусор → фолбэк. Реализация: `mlx/backend/metal/device_info.cpp` |

### Дренаж кэшей (`src/exo/worker/engines/mlx/patches/opt_batch_gen.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_CACHE_DRAIN_EVERY` | `auto` | пер-степовый `mx.eval(cache.state)`. auto: GLM-5.2 IndexShare — каждый шаг, DeepseekV4* — по AUTO_EVERY, прочим 0. `N` — форс всем, `off` — никому |
| `EXO_CACHE_DRAIN_AUTO_EVERY` | `1` | интервал дренажа для DSV4-ветки auto. Поднять до 8–32, если lm_bench покажет цену sync-стопа (запас по хэндлам гигантский) |
| `EXO_CACHE_DRAIN_ASYNC` | off | `1` — дренаж через `mx.async_eval` (без стопа конвейера) |
| `EXO_CACHE_DRAIN_WARN` | off | `1` — логировать проглоченные исключения дренажа |

Без ручки, всегда включено: **eval advance()-цепочек** (`_advance_chain_arrays`
в обеих ветках пер-степового async_eval) — закрывает механизм #4 для всех
моделей, стоимость ~0 (сотня крошечных int-опов асинхронно).

### Парковка префикс-кэшей (`src/exo/worker/engines/mlx/cache.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_PREFIX_CACHE_MAX_ENTRIES` | `32` | LRU-потолок запаркованных записей (≈35k буферов максимум). `0` — легаси-безлимит. Поднимать при >32 живых параллельных длинных сессий на инстансе |

Ручной сброс парковки без выгрузки инстанса: `kill -USR1 <pid раннера>`
(pid — самый жирный python; **только адресный kill**, у процессов без
хендлера USR1 = terminate). Сброс исполняется между шагами, в лог падает
`prefix-flush: dropped N parked cache entries`.

### Сторож/телеметрия (`src/exo/worker/engines/mlx/patches/resource_guard.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_MLX_CACHE_DRAIN_EVERY` | `512` | `mx.clear_cache()` раз в N шагов декода. `0` — выкл |
| `EXO_MLX_MEM_LOG_EVERY` | `0` | `N` — строка `[RESGUARD] step/drains/active/cache/peak` раз в N шагов. Включать при наблюдении |

### K3-диагностика (`src/exo/worker/engines/mlx/vendor/kimi_k3.py`)

| Ручка | Дефолт | Действие |
|---|---|---|
| `EXO_K3_NO_DECODE_COMPILE` | 0 | `1` — без `mx.compile(_decode_core)`. Бисекцией авг-2026 компайл оправдан; цена отключения ≈0 t/s на TP4 |
| `EXO_K3_NO_CONV_KERNEL` | 0 | `1` — conv через апстримный ShortConv1d вместо `k3_short_conv_step` |
| `EXO_K3_GD_*` | off | GD-кернелы (см. шапку файла; GPU Address Fault с RDMA — держать off) |

---

## 3. Мониторинг и приёмка

`mtl-count.sh` (ops-тулза, вне репо): счёт живых Metal-буферов.
`-i N` — системный (ioclasscount, мгновенно, безопасно);
`--heap PID` — точный пер-процессный (приостанавливает процесс на снимок).

Инварианты здорового кластера:
- декод любой модели: **коридор ±50–500, без монотонного наклона**;
- прогон 50 запросов: потолок ≈ база + 32×(буферов/запрос), дальше плато,
  в логах `evicted LRU entry ... due to entry cap`;
- `kill -USR1` в покое роняет счёт к базе.

Наклон = утечка: `Δ/с ÷ t/s` = буферов/токен; сверять с числом слоёв —
число само называет виновный слой.

---

## 4. Долги в апстрим

1. **ml-explore/mlx**: sysctl исчез, фолбэк 499000 консервативен ×8.4,
   стена 2²², мягкий отказ, кривая деградации вставки → просить
   `MLX_RESOURCE_LIMIT` (реализация готова: коммит в форке mlx).
2. **ml-explore/mlx-lm**: `ArraysCache.advance()` строит ленивые цепочки
   поверх массивов, не входящих в `state`, — невидимы для eval состояния;
   1 живой буфер/слой/токен в любом батч-потребителе. Воркэраунд — на
   стороне потребителя (см. `_advance_chain_arrays`); апстрим-фикс — eager
   в advance либо включить `lengths`/`left_padding` в `state`.

Доказательная база: `~/rsrc-wall-test/probe-*.log` (стена), логи mtl-count
прогонов lm_eval (утечки/фиксы).
