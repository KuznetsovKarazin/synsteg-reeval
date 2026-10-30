# Реальный bounded-Qwen pilot: 8 из 8 доставок, практическая схема ещё не завершена

25 октября 2026. Эксперимент `framed-interval-bounded-v1/native-pilot-v1`.

**Выполнены все 16 заранее заданных генераций и 8 свежих stego-получателей. Все 8 получателей аутентифицировали и точно восстановили 16-байтовый payload.** Независимый аудит сохранённых данных прошёл 89 009 проверок без отказов. Это результат транспорта на двух вымышленных источниках и четырёх закреплённых конфигурациях ключа, а не оценка надёжности на произвольных сообщениях, достоверности текста или скрытности.

Все 16 исходов сохранены. Генерации не повторялись, ключи и тексты не отбирались. Платных API, закрытого test и пользовательского ПК не было.

## Протокол и фактическая работа

Модель — прежний Qwen2.5-1.5B-Instruct GGUF Q4_K_M, llama.cpp b11791, те же native bytes, CPU backend, masked top50, temperature 0.8 и integer quantizer. Метод имеет новый ID: bounded domain/key derivation и одинаковая для stego/ordinary остановка на первом завершённом символе, разрешившем минимум 272 bits, либо на cap 768. Frame содержит 2 bytes длины, 16-byte payload slot и 16-byte AEAD tag. Получатель проверяет всю фактически переданную последовательность, включая surplus последнего символа.

Зафиксированы 2 prompt × 4 configuration × 2 arm = 16 sender. Сразу после каждого из 8 stego запускался отдельный свежий receiver. Ключ 32 bytes передавался отдельно; receiver не читает и не хеширует sender-fixtures. Parent сравнивает полученный plaintext с fixture только после терминального исхода receiver. Это проверенный маршрут чтений программы, а не ОС-sandbox.

Фактически выполнено **4942 sender token STEP**: 2388 stego и 2554 ordinary; дополнительно 2388 receiver replay STEP. Итого **7330 committed token STEP и 24 PROMPT requests**. Число низкоуровневых graph/forward invocations отдельно не измерялось. Все 24 Python worker завершились exit0 без worker/close errors. Process wrapper: exit0, 620.981437714001 s; отдельный OS exit native bridge не записывался и не подменяется статусом Python worker.

Порог достигнут во всех 16 случаях при 232–418 токенах, cap 768 нигде не понадобился. Resolved bits: 272–275; buffer lookahead 32 bits не является 32 потерянными битами ёмкости. Отказов, UNKNOWN или нулевых проводов в этой конкретной серии нет; полный протокол сохраняет такие исходы, если они возникают.

## Все 16 исходов

Слова считаются `len(text.split())`, без posthoc нормы длины. Compact envelope — ASCII canonical JSON без пробелов и перевода строки; он содержит public metadata, nonce и prompt IDs. Точная сохранённая pretty JSON версия больше и учтена отдельно.

| Случай | Токены | Слова | Wire, B | Envelope compact, B | Разрешённые биты | Результат |
|---|---:|---:|---:|---:|---:|---|
| station-c00-stego | 389 | 239 | 1458 | 2557 | 272 | 16 B доставлены |
| station-c00-ordinary | 418 | 212 | 1318 | 2560 | 275 | ordinary control |
| station-c01-stego | 302 | 195 | 1230 | 2557 | 273 | 16 B доставлены |
| station-c01-ordinary | 287 | 201 | 1322 | 2560 | 275 | ordinary control |
| station-c02-stego | 294 | 183 | 1173 | 2557 | 274 | 16 B доставлены |
| station-c02-ordinary | 328 | 185 | 1240 | 2560 | 274 | ordinary control |
| station-c03-stego | 299 | 182 | 1216 | 2557 | 273 | 16 B доставлены |
| station-c03-ordinary | 377 | 229 | 1475 | 2560 | 273 | ordinary control |
| library-c00-stego | 336 | 231 | 1360 | 2506 | 274 | 16 B доставлены |
| library-c00-ordinary | 313 | 213 | 1273 | 2509 | 273 | ordinary control |
| library-c01-stego | 275 | 234 | 1324 | 2506 | 272 | 16 B доставлены |
| library-c01-ordinary | 263 | 212 | 1259 | 2509 | 273 | ordinary control |
| library-c02-stego | 261 | 178 | 1064 | 2506 | 273 | 16 B доставлены |
| library-c02-ordinary | 288 | 221 | 1282 | 2509 | 274 | ordinary control |
| library-c03-stego | 232 | 179 | 1051 | 2506 | 273 | 16 B доставлены |
| library-c03-ordinary | 280 | 244 | 1373 | 2509 | 272 | ordinary control |


В каждой stego-строке first_threshold_token совпал с фактической длиной. Receiver обработал столько же токенов и проверил AEAD/fullwire. Ordinary здесь не имеет исхода payload delivery; его успешная остановка не включается в числитель 8/8.

## Стоимость и все восемь пар

| Показатель | Stego: среднее / медиана / диапазон | Ordinary: среднее / медиана / диапазон |
|---|---|---|
| Токены | 298.5 / 296.5 / 232–389 | 319.25 / 300.5 / 263–418 |
| Whitespace-слова | 202.625 / 189 / 178–239 | 214.625 / 212.5 / 185–244 |
| Wire, B | 1234.5 / 1223 / 1051–1458 | 1317.75 / 1300 / 1240–1475 |
| Envelope compact, B | 2531.5 / 2531.5 / 2506–2557 | 2534.5 / 2534.5 / 2509–2560 |
| Wire+envelope compact, B | 3766 / 3780 / 3557–4015 | 3852.25 / 3839 / 3768–4035 |


Всего передано 128 полезных bytes = 1024 bits в 9876 stego-wire bytes. Полезная скорость по stego-токенам **1024/2388 = 0.4288107203 bit/token**; по wire **1024/9876 = 0.1036857027 bit/byte**. Полезные bytes составляют 1.2960713% wire. Знаменатель содержит все 8 плановых stego-попыток; здесь все 8 доставили payload.

Compact envelopes для 8 stego составляют 20252 bytes, вместе с wire — 30128 bytes: 128/30128 = 0.42485396% полезных bytes. Сохранённые pretty envelopes — 32152 bytes. Это размеры текущей несжатой сериализации, а не доказанный минимум side information. Нельзя сообщать «достаточно 32 байтов ключа», скрывая эти расходы. Nonce 12 bytes уже представлен внутри envelope и второй раз к этому размеру не прибавляется. В fixture-матрице 4 разных 32-byte keys, то есть 128 bytes setup при передаче каждого ключа однажды; повторное использование ключа и публичные fixtures не доказывают production security. Одна общая модель занимает 1117320736 bytes; tokenizer/runtime/protocol входят в общие ресурсы и также не бесплатны, хотя их суммарный размер здесь не пересчитан.

Ниже каждая разница — stego минус ordinary одной пары; приведены **все 8 пар**.

| Пара | Δ токенов | Δ слов | Δ wire, B | Δ envelope, B | Δ wire+envelope, B |
|---|---:|---:|---:|---:|---:|
| station/c00 | -29 | +27 | +140 | -3 | +137 |
| station/c01 | +15 | -6 | -92 | -3 | -95 |
| station/c02 | -34 | -2 | -67 | -3 | -70 |
| station/c03 | -78 | -47 | -259 | -3 | -262 |
| library/c00 | +23 | +18 | +87 | -3 | +84 |
| library/c01 | +12 | +22 | +65 | -3 | +62 |
| library/c02 | -27 | -43 | -218 | -3 | -221 |
| library/c03 | -48 | -65 | -322 | -3 | -325 |


Средняя/медианная разница токенов: −20.75/−28, диапазон −78…+23. Для слов: −12/−4, диапазон −65…+27. Для wire: −83.25/−79.5 bytes, диапазон −322…+140. Разница envelope −3 bytes во всех парах объясняется длиной строкового имени arm, а не преимуществом метода. Это описательная сводка восьми детерминированных траекторий, без причинного вывода об экономии или IID-статистики.

## Точность и завершённость текстов

Независимый агентский posthoc review прочитал все 16 точных wire и проверил 184 span references. В выбранной разметке: 35 поддержанных источником наблюдений (32 прямых + 3 вычисленных), 27 неподтверждённых утверждений, 8 отдельно помеченных гипотез, 8 явных противоречий в 6 текстах и 2 неоднозначных случая. Противоречия отмечены в 5 stego и 1 ordinary; это **не human gold, не исчерпывающая разметка всех утверждений, не валидированная частота ошибок и не причинный эффект embedding**.

14 окончаний явно синтаксически неполны; ещё 2 могут быть завершёнными предложениями без финальной пунктуации. В 12 текстах имеются headings/lists вопреки заданному формату. Сохранён также `library-c03-ordinary`, где обзор не нашёл очевидного дополнительного дефекта основных фактов; у него остаются список и незавершённое окончание. Неоднозначные и нейтральные наблюдения не исключались.

Таким образом, **8/8 transport delivery не означает 8/8 качественных end-to-end результатов**. Пороговая остановка решает объём переданного frame, но сама по себе не завершает предложение и не удерживает факты источника.

## Что ещё мешает практическому применению

1. `envelope.protocol_sha256` сейчас равен SHA activation. Activation включает SHA `sender-fixtures.json`, а fixture-файл содержит ключи и закреплённые payload. Поэтому public binding связан с конкретным тестовым пакетом; изменение сообщения требует нового fixture/activation binding. Получатель не читает fixture, но **message-independent deployment ещё не реализован**. Этот pilot подтверждает доставку двух фиксированных prompt-specific payload, повторённых в четырёх конфигурациях, а не API для произвольных сообщений.
2. Имеются large shared resources, многокилобайтный public envelope и зависимость от точного native/tokenizer/runtime. Межмашинная воспроизводимость, compact production envelope и безопасный жизненный цикл ключей/nonce отдельно не доказаны.
3. Нет оценки обнаружимости всего текста. Условные integer laws и gate-аудит не заменяют стегоаналитический эксперимент; fullsoftmax mass вне masked top50 остаётся неизвестной.
4. Смысловая точность и естественное завершение текста остаются отдельной задачей. Новое правило конца должно быть заранее задано и воспроизводимо получателем; нельзя постфактум дописывать удобное окончание и считать его тем же wire.

Предыдущий fixed512 pilot дал 1/2 доставок; нынешний — 8/8. Одновременно поменялись ключи/nonce, domains, activation digest и stopping rule. **Это не контролируемая причинная оценка эффекта только stopping rule**, и две серии нельзя объединять в одну частоту надёжности.

## Воспроизводимость и сохранение

- Source freeze и фактический readback 27 dependencies предшествовали activation и inference.
- Activation SHA: `0a8e182e8e3144cb429acc6eaec638da2c3235e4c608f6facf76ef4fde70fe21`.
- Activation/intent/readback checkpoint: `5646910a024a4ec0e39be70ca1056107a454db55`.
- Primary package: 342 files, 24 endpoint slots, 25 ZIP_LZMA; fresh restore 342/342 exact. 16 readable UTF8 copies сохраняют wire bytes точно.
- Artifact index: `results/framed-interval-bounded-v1-artifacts/index.json`, SHA `57b28aa192865bc2c1c7807ff61d36da8f391e0c0d42ed60c5a73135d5ae3354`.
- **Git primary checkpoint: ожидается подтверждённый coordinator readback; root вставит commit после завершения сохранения.**
- Независимый audit: `results/framed-interval-bounded-v1-native-audit/independent-audit.json`; использовать числитель только при status=passed и failures=[].
- Качество: `results/framed-interval-bounded-v1-quality-review/native-pilot-v1.json`, SHA `fd1bdba9bc6d4c3922e3af9b90214d4a5ef6327fd458481cbb62b99cd5704b1a`.

Полные 16 строк, 8 paired costs, средние/медианы/диапазоны и input hashes находятся в `results/framed-interval-bounded-v1-native-metrics/`. Независимый арифметический аудит прошёл 752 проверки, включая все 16 строк, 8 пар и 106 входных SHA. Collector выполняет только арифметику по сохранённым worker/envelope/wire/audit, без NN, tokenizer или повторного decoder:

`python scripts/collect_framed_interval_bounded_v1_native_metrics.py --output <новая_директория>`

Он отказывается перезаписывать существующий output. Восстановление исходных 342 files:

`python scripts/package_framed_interval_bounded_v1_artifacts.py restore --destination <новая_директория>`

При очистке scratch сначала восстановить эти байты и прежние независимые audit/quality records из Git. Генерации, preflight и receiver inference не повторять.
