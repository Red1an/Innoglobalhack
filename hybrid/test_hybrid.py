import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid import stage2  # noqa: E402
from hybrid.cascade import run  # noqa: E402
from hybrid.normalize import normalize  # noqa: E402
from hybrid.router import RouterConfig, merge, route  # noqa: E402
from hybrid.triggers import detect  # noqa: E402


class FakeModel:
    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, texts):
        return [self.probs] * len(texts)


class FakeArbiter:
    def __init__(self, labels):
        self.labels, self.calls = labels, []

    def ask(self, text, mc, contested, triggers):
        self.calls.append(contested)
        return {"labels": self.labels, "cached": False, "seconds": 1.0, "input_tokens": 10, "output_tokens": 2,
                "error": None if self.labels is not None else "invalid_response"}


class NormalizeTest(unittest.TestCase):
    def test_homoglyphs_and_glue(self):
        self.assertEqual(normalize("штykaтypka по мaяkaм"), "штукатурка по маякам")
        self.assertEqual(normalize("1️⃣плитка2️⃣сантехника"), "; плитка ; сантехника")
        self.assertEqual(normalize("ШтукатуркаПлитка​"), "Штукатурка Плитка")

    def test_latin_words_untouched(self):
        self.assertEqual(normalize("Knauf MP-75, 10м2"), "Knauf MP-75, 10м2")


class TriggerTest(unittest.TestCase):
    def test_negation_targets_object(self):
        self.assertEqual(detect("электрику и сантехнику не выполняем")["negation"], {104, 105})
        self.assertNotIn("negation", detect("не нужно искать плиточника, работаем без посредников"))

    def test_seo_tail_only_new_classes(self):
        self.assertEqual(detect("ремонт квартир. #штукатурка #плитка")["seo_tail"], {102, 103})
        self.assertNotIn("seo_tail", detect("штукатурка стен. #штукатурка"))

    def test_warm_floor_typed(self):
        self.assertIn("warm_floor", detect("монтаж тёплого пола"))
        self.assertNotIn("warm_floor", detect("электрический тёплый пол под плитку"))


class RouterTest(unittest.TestCase):
    cfg = RouterConfig(0.2, 0.8, 0.5)

    def test_freeze_and_contest(self):
        r = route([0.05, 0.95, 0.5, 0.9, 0.1], 101, "укладка плитки", self.cfg)
        self.assertEqual(r.frozen, {102: 0, 103: 1, 105: 1, 106: 0})
        self.assertEqual(r.contested, [104])

    def test_llm_cannot_change_frozen(self):
        r = route([0.05, 0.95, 0.5, 0.9, 0.1], 101, "укладка плитки", self.cfg)
        self.assertEqual(merge(r, [102, 104, 106]), [103, 104, 105])
        self.assertEqual(merge(r, []), [103, 105])

    def test_trigger_blocks_high_freeze(self):
        r = route([0.05, 0.05, 0.05, 0.95, 0.05], 101, "электрику не выполняем", self.cfg)
        self.assertEqual(r.contested, [105])

    def test_source_excluded_and_101_check(self):
        r = route([0.9, 0.9, 0.05, 0.05, 0.05], 102, "ремонт квартиры под ключ, укладка плитки", self.cfg)
        self.assertNotIn(102, r.frozen)
        self.assertIn(101, r.contested)
        r = route([0.9, 0.9, 0.05, 0.05, 0.05], 102, "укладка плитки", self.cfg)
        self.assertEqual(r.frozen[101], 0)


class CascadeTest(unittest.TestCase):
    rows = [{"row_id": 1, "mcId": 101, "description": "укладка плитки"}]

    def test_only_contested_sent(self):
        arb = FakeArbiter([104, 102])
        rec = run(self.rows, FakeModel([0.05, 0.95, 0.5, 0.9, 0.1]), RouterConfig(0.2, 0.8), "cascade", arb)[0]
        self.assertEqual(arb.calls, [[104]])
        self.assertEqual(rec["labels"], [103, 104, 105])

    def test_fallback_to_stage1(self):
        rec = run(self.rows, FakeModel([0.05, 0.95, 0.6, 0.9, 0.1]), RouterConfig(0.2, 0.8), "cascade", FakeArbiter(None))[0]
        self.assertTrue(rec["fallback"])
        self.assertEqual(rec["labels"], [103, 104, 105])

    def test_confident_rows_skip_llm(self):
        arb = FakeArbiter([102])
        rec = run(self.rows, FakeModel([0.05, 0.95, 0.05, 0.9, 0.1]), RouterConfig(0.2, 0.8), "cascade", arb)[0]
        self.assertEqual(arb.calls, [])
        self.assertEqual(rec["labels"], [103, 105])


class PromptTest(unittest.TestCase):
    def test_only_contested_categories_and_rules(self):
        p = stage2.build_prompt([102], ["leveling_no_plaster"], "off")
        self.assertIn("102 Штукатурные", p)
        self.assertNotIn("104 Сантехника", p)
        self.assertIn("Общее выравнивание стен, в том числе шпаклёвкой", p)
        self.assertNotIn("хэштегах", p)
        self.assertIn("не подтверждённая", stage2.build_prompt([102], ["seo_tail"], "off"))
        self.assertIn("считается предложенной", stage2.build_prompt([102], [], "on"))

    def test_parse(self):
        self.assertEqual(stage2.parse('{"labels": [104, 102, 105]}', [102, 104]), [102, 104])
        self.assertEqual(stage2.parse('```json\n{"labels": []}\n```', [102]), [])
        self.assertIsNone(stage2.parse("нет", [102]))

    def test_cache_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            arb = stage2.OllamaArbiter(cache_dir=folder)
            reply = {"message": {"content": '{"labels": [104]}'}, "prompt_eval_count": 100, "eval_count": 5}
            with mock.patch.object(arb, "_post", return_value=reply) as post:
                first = arb.ask("текст", 101, [104], [])
                second = arb.ask("текст", 101, [104], [])
            self.assertEqual(post.call_count, 1)
            self.assertEqual((first["labels"], first["cached"]), ([104], False))
            self.assertEqual((second["labels"], second["cached"]), ([104], True))
            with mock.patch.object(arb, "_post", side_effect=OSError):
                self.assertIsNone(arb.ask("другой", 101, [104], [])["labels"])
            self.assertEqual(len(list(Path(folder).rglob("*.json"))), 1)

    def test_schema_limits_repetition(self):
        with tempfile.TemporaryDirectory() as folder:
            arb = stage2.OllamaArbiter(cache_dir=folder)
            reply = {"message": {"content": '{"labels": []}'}, "prompt_eval_count": 10, "eval_count": 3}
            with mock.patch.object(arb, "_post", return_value=reply) as post:
                arb.ask("текст", 101, [102, 104], [])
            labels = post.call_args[0][1]["format"]["properties"]["labels"]
            self.assertEqual((labels["maxItems"], labels["items"]["enum"]), (2, [102, 104]))


class SplitTest(unittest.TestCase):
    def test_same_train_fit_as_stress_suite_and_no_leaks(self):
        from eval_stress_suite import build_leak_free_train
        from hybrid.data import GOLD, STRESS, TRAIN, read_jsonl, reserved_for_eval, split_train
        from bench import gold_overlap_key, key
        train_fit, dev = split_train(reserved_rows=reserved_for_eval())
        ref = build_leak_free_train(read_jsonl(STRESS), TRAIN, GOLD)
        ours = {key(r["description"]) for r in train_fit}
        self.assertEqual(ours, {key(r["description"]) for r in ref})
        norm = gold_overlap_key()
        held = {norm(r["description"]) for r in reserved_for_eval()}
        self.assertFalse({norm(r["description"]) for r in train_fit} & held)
        self.assertFalse(ours & {key(r["description"]) for r in dev})
        dev_items = {r["item_id"] for r in dev}
        import pyarrow.parquet as pq
        src = {r["item_id"]: r for r in pq.read_table(TRAIN).to_pylist()}
        synth_sources = {src[r["item_id"]]["source_item_id"] for r in train_fit if r["origin"] == "synthetic"}
        self.assertFalse(synth_sources & dev_items)


if __name__ == "__main__":
    unittest.main()
