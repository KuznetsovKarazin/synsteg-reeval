"""Validate and assemble manual posthoc annotations; no quality labels inferred."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID = "prospective-development-transport-first40-delta-slot-review-v1"
OUT = ROOT / "results" / ID
LABELS = {"meaning_change", "grammar_or_pos", "style_or_collocation", "ambiguous", "no_obvious_added_defect"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines() if line.strip()]


def lines(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows)


def validate_span(text, span):
    assert text[span["start"]:span["end"]] == span["quote"]


def build():
    inventory = json.loads((OUT / "inventory.json").read_text())
    for name, digest in inventory["artifact_sha256"].items():
        assert sha((OUT / name).read_bytes()) == digest
    for item in inventory["frozen_inputs"]:
        assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"], item["path"]
    slots, covers, outcomes = load("unique_slots.jsonl"), load("covers.jsonl"), load("outcomes.jsonl")
    slot_by_id = {row["slot_id"]: row for row in slots}
    cover_by_id = {row["record_id"]: row for row in covers}
    outcome_by_id = {row["row_id"]: row for row in outcomes}
    annotation_by_id = {}
    input_artifacts = []
    for part, lower, upper in [("a", 31, 33), ("b", 34, 36), ("c", 37, 40)]:
        path = OUT / f"review-part-{part}.jsonl"
        data = load(path.name)
        input_artifacts.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), "rows": len(data)})
        expected_ids = {row["slot_id"] for row in slots if lower <= row["source_order"] <= upper}
        assert {row["slot_id"] for row in data} == expected_ids
        for row in data:
            key = row["slot_id"]
            assert key not in annotation_by_id
            slot = slot_by_id[key]
            for field in ["source_order", "token_index_zero_based", "original", "replacement"]:
                assert row[field] == slot[field], (key, field)
            labels = row["labels"]
            assert labels and len(labels) == len(set(labels)) and set(labels) <= LABELS
            assert "no_obvious_added_defect" not in labels or labels == ["no_obvious_added_defect"]
            assert row["review_status"] == "read_all_associated_contexts" and len(row["reason_ru"]) >= 20
            all_ids = [item["row_id"] for item in slot["occurrences"]]
            assert len(row["reviewed_occurrence_row_ids"]) == len(all_ids)
            assert set(row["reviewed_occurrence_row_ids"]) == set(all_ids)
            assert len(row["issue_row_ids"]) == len(set(row["issue_row_ids"]))
            assert set(row["issue_row_ids"]) <= set(all_ids)
            assert (not row["issue_row_ids"]) == (labels == ["no_obvious_added_defect"])
            annotation_by_id[key] = row
    assert len(annotation_by_id) == len(slots) == 180
    annotated = []
    for slot in slots:
        cover = cover_by_id[slot["record_id"]]
        validate_span(cover["cover_text"], slot["original_span"])
        validate_span(cover["cover_text"], slot["original_context"])
        for occurrence in slot["occurrences"]:
            outcome = outcome_by_id[occurrence["row_id"]]
            assert outcome["record_id"] == slot["record_id"]
            assert sha(outcome["stego_text"].encode()) == occurrence["stego_sha256"]
            validate_span(outcome["stego_text"], occurrence["stego_span"])
            validate_span(outcome["stego_text"], occurrence["stego_context"])
        annotated.append({**slot, "annotation": annotation_by_id[slot["slot_id"]]})
    outcome_reviews = []
    for outcome in outcomes:
        key = outcome["row_id"]
        ids = outcome["changed_slot_ids"]
        issue_ids = [slot_id for slot_id in ids if key in annotation_by_id[slot_id]["issue_row_ids"]]
        labels = sorted({label for slot_id in issue_ids for label in annotation_by_id[slot_id]["labels"]})
        outcome_reviews.append({"row_id": key, "record_id": outcome["record_id"], "source_order": outcome["source_order"], "parent_payload_length": outcome["parent_payload_length"], "requested_payload_bits": outcome["requested_payload_bits"], "arm": outcome["arm"], "execution_status": outcome["execution_status"], "stego_sha256": outcome["stego_sha256"], "changed_slot_occurrences": len(ids), "all_changed_slot_ids": ids, "observed_issue_or_ambiguity_slot_ids": issue_ids, "observed_labels_present": labels, "review_complete_for_changed_slots": True, "validated_quality_pass_fail": None})
    cover_reviews = []
    for cover in covers:
        local_slots = [row for row in annotated if row["record_id"] == cover["record_id"]]
        local_outcomes = [row for row in outcome_reviews if row["record_id"] == cover["record_id"]]
        cover_reviews.append({"source_order": cover["source_order"], "record_id": cover["record_id"], "title": cover["title"], "cover_sha256": cover["cover_sha256"], "source_excerpt_sha256": cover["source_excerpt_sha256"], "all_cell_execution_status_counts": cover["all_cell_execution_status_counts"], "successful_stego_outcomes": len(local_outcomes), "changed_slot_occurrences": sum(row["changed_slot_occurrences"] for row in local_outcomes), "unique_changed_slots": len(local_slots), "reviewed_unique_changed_slots": len(local_slots), "status": "all_observed_substitutions_reviewed" if local_slots else "no_successful_stego_no_substitution_review", "observed_label_counts_unique_slots": dict(sorted(Counter(label for row in local_slots for label in row["annotation"]["labels"]).items())), "validated_quality_pass_fail": None})
    with gzip.open(ROOT / inventory["milestone"] / "cells.jsonl.gz", "rt") as stream:
        cells = [json.loads(line) for line in stream if line.strip()]
    attempt_rows = [{**{key: cell[key] for key in ["row_id", "record_id", "source_order", "arm", "grid", "parent_payload_length", "requested_payload_bits", "execution_status", "full_exact", "changed_positions"]}, "substitution_review_status": "all_changed_slots_reviewed" if cell["full_exact"] is True else "not_applicable_no_successful_stego"} for cell in cells if cell.get("encoder_called") is True and cell["model_alias"] == "tinyllama11" and 31 <= cell["source_order"] <= 40]
    assert len(attempt_rows) == 240 and len(outcome_reviews) == 24 and len(cover_reviews) == 10
    labels_unique = Counter(label for row in annotated for label in row["annotation"]["labels"])
    labels_occurrences = Counter()
    for row in annotated:
        n = len(row["annotation"]["issue_row_ids"])
        if row["annotation"]["labels"] == ["no_obvious_added_defect"]:
            n = len(row["occurrences"])
        for label in row["annotation"]["labels"]:
            labels_occurrences[label] += n
    summary = {"schema": "synsteg-posthoc-slot-review-summary-v1", "id": ID, "inventory_sha256": sha((OUT / "inventory.json").read_bytes()), "review_parts": input_artifacts, "code_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in [Path(__file__), Path(__file__).with_name("build_inventory.py")]}, "generation_calls": 0, "encoder_calls": 0, "receiver_calls": 0, "planned_model_record_rows": 300, "completed_covers": 10, "cumulative_completed_covers": 40, "source_orders_reviewed": list(range(31, 41)), "prior_full_exact_outcomes_not_reannotated": 75, "prior_encoder_attempts_reused_not_reviewed_again": 720, "full_cumulative_planned_cells": 15300, "actual_encoder_attempts": 240, "insufficient_capacity": 216, "full_exact_outcomes": 24, "reviewed_full_exact_outcomes": 24, "changed_slot_occurrences": 380, "reviewed_changed_slot_occurrences": sum(len(row["annotation"]["reviewed_occurrence_row_ids"]) for row in annotated), "unique_changed_slots": 180, "reviewed_unique_changed_slots": len(annotated), "unreviewed_slots": [], "successful_cover_orders": inventory["successful_cover_orders"], "cover_orders_without_successful_stego": inventory["cover_orders_without_successful_stego"], "successful_arm_counts": dict(Counter(row["arm"] for row in outcomes)), "successful_requested_payload_bits_counts": dict(Counter(row["requested_payload_bits"] for row in outcomes)), "observed_label_counts_unique_slots": dict(sorted(labels_unique.items())), "observed_label_counts_occurrences": dict(sorted(labels_occurrences.items())), "outcomes_with_at_least_one_recorded_issue_or_ambiguity": sum(bool(row["observed_issue_or_ambiguity_slot_ids"]) for row in outcome_reviews), "human_gold": False, "blind_review": False, "inter_rater_agreement_measured": False, "numeric_factuality_or_security_metric": None, "labels_are_overlapping_descriptive_agent_observations": True, "limitations_ru": ["Проверены все фактические замены всех 24 новых успешных исходов, без выбора удачных примеров. 216 новых отказов по ёмкости не получают оценку качества stego; все 10 новых covers остаются в знаменателе.", "Уникальный ключ объединяет только точные record_id, token index, original и replacement; каждый row_id, точный stego-контекст и его offsets сохранены. Разные окружающие замены рассмотрены в фактических вариантах.", "Это постфактум агентское описание добавленного эффекта замены, не human gold, не слепой эксперимент и не валидированная численная фактичность/безопасность. Части размечались разными агентами без измерения согласия; метки могут пересекаться.", "no_obvious_added_defect означает только отсутствие очевидного дополнительного дефекта данной локальной замены при чтении. Это не доказательство синонимии, качества всего текста или безопасности метода.", "Исходные covers уже содержат неподтверждённые excerpt сведения; их наличие не приписывается кодеку заново. Новая замена может дополнительно портить смысл даже в ранее дефектном предложении.", "Все 24 новых успеха относятся к public_v1_plain_length16 и полезной нагрузке 16 бит; это повторные клетки трёх родительских payload на 8 новых covers, а не 24 независимых генерации.", "Подсчёты меток — проверяемые свойства этой сохранённой разметки, не частоты объективных ошибок или популяционная оценка. Внешняя истинность исторических/научных сведений не проверялась."]}
    assert summary["reviewed_changed_slot_occurrences"] == 380
    data = {"annotated_unique_slots.jsonl": lines(annotated), "outcome_reviews.jsonl": lines(outcome_reviews), "cover_reviews.jsonl": lines(cover_reviews), "all_executed_attempts.jsonl": lines(attempt_rows), "summary.json": (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()}
    report = ["# Posthoc обзор новых замен: TinyLlama №31–40", "", "Проверены все **24 новых full_exact исхода**, все **380 фактических замен** и **180 уникальных слотов**. Знаменатель — **240 новых encoder attempts на всех 10 новых covers**; **216 insufficient_capacity** сохранены. Все новые доставки — `public_v1_plain_length16`, полезная нагрузка 16 бит. Прежние 720 условий и 75 успешных исходов first30 не размечались повторно.", "", "Это описательная агентская разметка добавленного эффекта замены, не human gold, не слепая оценка и не валидированная частота ошибок. Межэкспертное согласие не измерялось. Уже существовавшие неподтверждённые сведения covers не приписываются кодеку. Все фактические варианты окружающих замен прочитаны в назначенных группах.", "", "## Полнота", "", "| Cover | Source | Успешных stego | Замен | Уникальных слотов |", "|---:|---|---:|---:|---:|"]
    for row in cover_reviews:
        report.append(f"| {row['source_order']} | {row['title'].replace('|', '/')} | {row['successful_stego_outcomes']} | {row['changed_slot_occurrences']} | {row['unique_changed_slots']} |")
    report += ["", "№37 и №40 остаются в знаменателе без успешного stego. Нулевая строка не является положительной оценкой качества cover. Все 240 новых выполненных условий находятся в `all_executed_attempts.jsonl`; полный cumulative grid 15 300 клеток и 300 model-record pairs остаётся в исходном milestone.", "", "## Описательные метки", "", "Метки пересекаются; следующие числа относятся только к сохранённой разметке, не к объективной или популяционной частоте ошибок.", "", "| Метка | Уникальных слотов | Вхождений |", "|---|---:|---:|"]
    for label in sorted(labels_unique):
        report.append(f"| {label} | {labels_unique[label]} | {labels_occurrences[label]} |")
    report += ["", "`no_obvious_added_defect` означает только отсутствие очевидного дополнительного дефекта локальной замены; `ambiguous` сохраняет неопределённость. Эти метки не сертифицируют синонимию, весь текст или безопасность.", "", "## Примеры из полного набора", "", "Примеры поясняют разметку; все 180 слотов, включая приемлемые и неоднозначные, сохранены с точными offsets, SHA и crosswalk всех 24 исходов.", "", "| Cover / token | Замена | Наблюдение |", "|---|---|---|"]
    examples = [(34,"Chase","Dog"),(35,"Technology","Engineering"),(35,"chip","fleck"),(38,"potential","potentiality"),(38,"three","trio"),(39,"terms","footing"),(38,"calculate","compute"),(39,"halfway","midway")]
    for order, original, replacement in examples:
        row = next(row for row in annotated if (row["source_order"],row["original"],row["replacement"]) == (order,original,replacement))
        report.append(f"| {order} / {row['token_index_zero_based']} | {original} → {replacement} | {row['annotation']['reason_ru'].replace('|','/')} |")
    report += ["", f"Сохранённые замечания/неоднозначности отмечены хотя бы для одного слота в {summary['outcomes_with_at_least_one_recorded_issue_or_ambiguity']} из 24 новых исходов; это свойство данной агентской разметки, не валидированная частота ошибок. Механическая доставка 16 бит сама по себе не подтверждает сохранение смысла: повреждаются технические термины, собственные имена, числительные и синтаксические конструкции. Внешняя истинность научных/исторических утверждений здесь не проверялась.", "", "## Воспроизведение без codec", "", "`build_inventory.py --verify-existing` повторно извлекает наблюдаемые замены из сохранённых jobs/snapshots и сравнивает inventory побайтово. `compile_review.py --verify-existing` проверяет ручные части, все spans, SHA и полные связи outcomes/slots, затем сравнивает итоговые bytes. Encoder, receiver, tokenizer или модель не вызываются. Ручные языковые суждения остаются входными данными, а не алгоритмически доказанными выводами."]
    data["report_ru.md"] = ("\n".join(report) + "\n").encode()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    data = build()
    for name, content in data.items():
        path = OUT / name
        if args.verify_existing:
            assert path.read_bytes() == content, name
        else:
            with path.open("xb") as stream:
                stream.write(content)
    print(json.dumps({"mode": "verified" if args.verify_existing else "created", "outputs": {name: sha(content) for name, content in data.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
