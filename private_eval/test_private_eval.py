import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from private_eval import fmt  # noqa: E402
from private_eval.drafts import extract, parse_draft  # noqa: E402
from private_eval.llm import parse_labels  # noqa: E402


class FormatTest(unittest.TestCase):
    def test_writer_reproduces_organizer_example(self):
        gold = HERE / "examples" / "example.gold.csv"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.csv"
            fmt.write_answer(out, fmt.read_answer(gold))
            self.assertEqual(out.read_bytes(), gold.read_bytes())

    def test_check_catches_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.csv"
            fmt.write_answer(out, [{"testcase": "x", "labels": [104, 101], "descriptions": ["a"]}])
            problems = fmt.check_answer(out)
            self.assertTrue(any("по возрастанию" in p for p in problems))
            self.assertTrue(any("исходную" in p for p in problems))
            self.assertTrue(any("длине" in p for p in problems))

    def test_example_gold_is_valid(self):
        self.assertEqual(fmt.check_answer(HERE / "examples" / "example.gold.csv", HERE / "examples" / "example.csv"), [])


class ParseTest(unittest.TestCase):
    def test_labels(self):
        allowed = [102, 103, 104, 105, 106]
        self.assertEqual(parse_labels('```json\n{"labels": [104, 102, 104]}\n```', allowed), [102, 104])
        self.assertEqual(parse_labels('Ответ: {"labels": [101, 105]}', allowed), [105])
        self.assertIsNone(parse_labels('{"labels": ["102"]}', allowed))
        self.assertIsNone(parse_labels(None, allowed))

    def test_draft(self):
        self.assertEqual(parse_draft("Заголовок: Штукатурка\nТекст:\nШтукатурим стены."), ("Штукатурка", "Штукатурим стены."))
        self.assertEqual(parse_draft("Просто текст."), (None, "Просто текст."))

    def test_extract_keeps_target_and_drops_other_services(self):
        text = ("Штукатурим стены по маякам в квартирах. Также меняем смесители и сифоны. "
                "Приезжаем по предварительной записи. Звоните +7 999 123-45-67.")
        draft = extract(text, 102)
        self.assertIn("маякам", draft)
        self.assertIn("предварительной записи", draft)
        self.assertNotIn("смесители", draft)
        self.assertNotIn("999", draft)


if __name__ == "__main__":
    unittest.main()
