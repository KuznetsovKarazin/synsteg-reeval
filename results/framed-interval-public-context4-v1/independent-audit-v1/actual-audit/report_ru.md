# Независимый аудит публичного генеративного канала: четыре контекста

**Подтверждены 4/4 точные аутентифицированные передачи по 16 байт.** Все 12 запланированных endpoint-процессов завершились и прошли аудит: 4 stego sender, 4 ordinary sender и 4 свежих receiver. Отказов, неизвестных исходов, дополнительных попыток и исключений из знаменателя в этой кампании нет. Это ограниченный результат доставки на четырёх development-контекстах, одном ключе и одном runtime.

Источником служат [сохранённые raw artifacts](https://github.com/KuznetsovKarazin/synsteg-independent-reproduction/tree/2a3be59b6a808464837156753d97f7278880cbcd/artifacts/framed-interval-public-context4-v1/raw-v1), commit `2a3be59b6a808464837156753d97f7278880cbcd`. Проверка выполнена один раз на свежем восстановлении из Git. Аудитор не импортирует исполняемые модули проекта: использует собственные parser/hash-chain, квантование, HMAC ordering и integer transitions, стандартную библиотеку Python и cryptography. Вызовов модели, native bridge или tokenizer во время аудита: **0**. Проверка смысла и фактической верности текстов делегирована отдельному аудитору и в этот вердикт не входит.

## Что независимо проверено

- Git blob SHA двух archive index и source closure перечитаны непосредственно из Git; локальные байты совпали. Обе activation и оба source-readback также перечитаны непосредственно из соответствующих commit. Source freeze является предком primary commit (Git compare: ahead 11, behind 0).
- Обе ZIP восстановлены и проверены по полному содержимому: **412 файлов результатов, 106847399 B, 55 директорий**, плюс **8 nonce-файлов, 1864 B**. Проверены все archive parts, archive SHA256, membership, точные байты каждого файла и отсутствие лишних/потерянных файлов или директорий. Это повторная проверка сохранённых bytes, а не повторный эксперимент.
- Все **30 файлов source closure** совпали. Финальный helper: `b97aa4ac71f6a2c6971ae72460d3b65275eff4db1539b34c05e39fb5c32367af`. Обе activation привязаны к этому closure и предложению; все source/tokenization hashes из gate proofs проверены.
- Четыре подготовительных context TOKENIZE сохранены до execute activation. Prompt IDs содержат **211, 202, 203, 246** токенов для orders 1, 4, 7, 10. Подготовка заявляет TOKENIZE=4 и PROMPT/STEP=0; дополнительные tokenizer-команды внутри endpoint-кодека в это число не входят.
- Все 12 supervisor records подтверждают exit 0, соблюдение 300-секундного endpoint budget и завершение группы до reap. Сохранены **12 разных native process identities**. Восьми sender соответствуют **8 разных сожжённых key/nonce reservations**; receipts совпадают с полным ledger. Root coordinator завершился с exit 0 за **614.228 s**.
- Перепроверены **2572 token transitions / 5144 journal events / 128600 сохранённых candidate-logit строк**. Цепочки prepare/commit, byte hashes, prefix IDs, quantization из сохранённых float32 logits, admission weights, keyed ordering, integer bins, TV, before/after state, emitted bits и выбор токена по входному bitstream совпали. Не восстановлены заново сами logits модели и canonical tokenizer eligibility.
- Для четырёх свежих receiver полный wire, token IDs и traces совпали с sender. Восстановлены первые 272 бита/34 байта, отдельно выполнен AEAD-decrypt; все 16 байт равны `000102030405060708090a0b0c0d0e0f`. Первый порог >=272 достигается ровно на последнем токене, лишнего wire после него нет. Избыточные resolved bits относятся к хвосту bitstream и не объявлены дополнительным payload.

## Вызовы текущей кампании

| Этап | Native loads | PROMPT | STEP | DIST | TOKENIZE |
|---|---:|---:|---:|---:|---:|
| Подготовка четырёх контекстов | 4 | 0 | 0 | 0 | 4 |
| 12 endpoints | 12 | 12 | 2572 | 2572 | 116967 |
| Всего в новом опыте | 16 | 12 | 2572 | 2572 | 116971 |

Это счётчики только этой кампании, не исторических запусков. Для endpoints counts восстановлены по frozen source и полным успешным journals; отдельно записанного native command counter нет. В частности, TOKENIZE=116967 — **source/journal-derived**, не прямая native telemetry. В расчёте: два gate constructors на endpoint; для каждого token две prefix validations и selected-token reinspection; encode для каждого candidate с verdict accepted/retokenization_mismatch; один fullwire encode на receiver. invalid_utf8 отвергается Python decode до native TOKENIZE. PROMPT обработали суммарно 2586 входных токенов. Во время самого аудита и подсчёта новые NN/TOKENIZE вызовы отсутствуют. Полное разложение — `command-counts.json`.

## Полные знаменатели и цена передачи

| Случай | Токены wire | Resolved bits | Wire, B | Envelope, B | Context, B | Wire + envelope, B | Payload |
|---|---:|---:|---:|---:|---:|---:|---|
| o001-stego | 190 | 287 | 986 | 352 | 1126 | 1338 | 16 B, точно |
| o001-ordinary | 224 | 276 | 1203 | 355 | 1126 | 1558 | контроль |
| o004-stego | 207 | 274 | 1181 | 352 | 1069 | 1533 | 16 B, точно |
| o004-ordinary | 227 | 283 | 1306 | 355 | 1069 | 1661 | контроль |
| o007-stego | 151 | 273 | 768 | 352 | 1082 | 1120 | 16 B, точно |
| o007-ordinary | 229 | 272 | 1192 | 355 | 1082 | 1547 | контроль |
| o010-stego | 264 | 275 | 1283 | 352 | 1240 | 1635 | 16 B, точно |
| o010-ordinary | 268 | 279 | 1360 | 355 | 1240 | 1715 | контроль |

Все восемь генераций вошли в таблицу. Размеры envelope/context здесь равны фактическим compact JSON файлам. Ordinary — независимые публичные биты через тот же keyed transform; это не свободный sampling модели и не payload-доставка.

Итого **64 B payload → 4218 B wire**, или **5626 B с четырьмя envelope** при уже известных context/spec. Это **0.6305 payload bit/token**, **1.138%** полезных байтов в wire+envelope. Четыре различных public context занимают ещё **4517 B**. При однократной передаче compact spec **1188 B** совокупная стоимость четырёх сообщений с этими контекстами — **11331 B**, до profile, ключа и runtime/model setup. Сумма четырёх независимых first-message costs в `audit.json` равна 14895 B и намеренно считает spec четыре раза; это отдельный сценарий учёта.

Кадр каждого сообщения: **16 B payload + 18 B framing/tag = 34 B**. Ключ: **32 B** заранее разделённого состояния. Публичный nonce: **12 B**, уже включён в envelope, второй раз не добавляется. Profile: **14747 B файл / 12747 B compact JSON**. Spec: **1388 B файл / 1188 B compact JSON**. Зарегистрированные **41 именованный ресурс — 1213380452 B**, из них модель — **1117320736 B**. Сумма по именам включает архив, варианты и aliases библиотек; она не равна уникальным inode и не включена в приведённую долю payload.

## Контракт и границы результата

Receiver получает wire + 32-byte key + public envelope/context/spec/profile + доверенные ресурсы. Проверены argv и точные файлы inputs: нет payload-file, nonce-ledger, fixture либо sender trace. Inventory отдельной публичной установки содержит 61 файл и совпадает с исходниками, четырьмя public contexts и 41 resource identity. Запуск через exec очищает память координатора, но это **логическое разделение на одном хосте, без OS sandbox**.

Бинарные model/runtime файлы аудитор заново не загружал и не хешировал: проверены сохранённые resource/preprompt/resident hashes и их согласованность с frozen profile. Это не новая аппаратная аттестация и не межхостовое воспроизведение. Ключ и payload являются публичными test fixtures. Envelope явно раскрывает arm; четыре доставки не доказывают скрытность, практическую безопасность, естественность или фактическую верность текста. Отдельно остаются естественное окончание текста и способ доставки публичного metadata/context. Новое ограничение 300 s отличается от старого 1800 s; числа старого пилота не объединяются с этими как одинаковый режим.

Машиночитаемые доказательства: `audit.json`, `derived-metrics.json`; исполняемый проверяющий скрипт: `scripts/audit_public_context4_postgit_v1.py`. Исходные журналы и результаты не изменялись. `actual-audit.stdout.log` фиксирует единственный полный вызов: exit 0, 12 endpoints, 4 exact, blockers=[]; stderr пуст.
