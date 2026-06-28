import csv
import tempfile
import unittest
from pathlib import Path


class InspectDataTests(unittest.TestCase):
    def test_summarize_csv_reports_schema_and_missing_values(self):
        from src.inspect_data import summarize_table_file

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "train.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "Tag", "Severity"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "1", "Tag": "Access Control", "Severity": "High"},
                        {"id": "2", "Tag": "Reentrancy", "Severity": "Medium"},
                        {"id": "3", "Tag": "", "Severity": "High"},
                    ]
                )

            summary = summarize_table_file(csv_path)

        self.assertEqual(summary["shape"], [3, 3])
        self.assertEqual(summary["columns"], ["id", "Tag", "Severity"])
        self.assertEqual(summary["missing_ratio"]["Tag"], 1 / 3)
        self.assertEqual(summary["unique_count"]["Severity"], 2)
        self.assertEqual(summary["head"][0]["Tag"], "Access Control")

    def test_inspect_empty_raw_directory_marks_data_blocked(self):
        from src.inspect_data import inspect_raw_data

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "data" / "raw_kaggle"
            report_dir = tmp_path / "outputs" / "reports"
            working_dir = tmp_path / "working"
            data_dir = tmp_path / "data"
            raw_dir.mkdir(parents=True)

            summary = inspect_raw_data(
                raw_dir=raw_dir,
                data_dir=data_dir,
                report_dir=report_dir,
                working_dir=working_dir,
            )

            report_text = (report_dir / "data_report.md").read_text(encoding="utf-8")
            schema_text = (working_dir / "schema_summary.json").read_text(encoding="utf-8")
            file_list_text = (data_dir / "file_list.txt").read_text(encoding="utf-8")

        self.assertEqual(summary["status"], "blocked_missing_kaggle_data")
        self.assertEqual(summary["files"], [])
        self.assertIn("data/raw_kaggle is empty", report_text)
        self.assertIn("Kaggle CLI", report_text)
        self.assertIn("blocked_missing_kaggle_data", schema_text)
        self.assertEqual(file_list_text.strip(), "")

    def test_detect_sample_submission_and_target_columns(self):
        from src.inspect_data import inspect_raw_data

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "data" / "raw_kaggle"
            report_dir = tmp_path / "outputs" / "reports"
            working_dir = tmp_path / "working"
            data_dir = tmp_path / "data"
            raw_dir.mkdir(parents=True)

            self._write_rows(
                raw_dir / "sample_submission.csv",
                ["id", "Tag", "Description"],
                [{"id": "a", "Tag": "", "Description": ""}, {"id": "b", "Tag": "", "Description": ""}],
            )
            self._write_rows(
                raw_dir / "test.csv",
                ["id", "source"],
                [{"id": "a", "source": "contract A {}"}],
            )
            self._write_rows(
                raw_dir / "train.csv",
                ["id", "source", "Tag", "Description"],
                [
                    {
                        "id": "x",
                        "source": "contract B {}",
                        "Tag": "Access Control",
                        "Description": "Missing access control.",
                    }
                ],
            )

            summary = inspect_raw_data(
                raw_dir=raw_dir,
                data_dir=data_dir,
                report_dir=report_dir,
                working_dir=working_dir,
            )

        sample = summary["sample_submission"]
        self.assertTrue(sample["path"].endswith("sample_submission.csv"))
        self.assertEqual(sample["columns"], ["id", "Tag", "Description"])
        self.assertEqual(sample["id_column"], "id")
        self.assertEqual(sample["target_columns"], ["Tag", "Description"])
        self.assertTrue(summary["identified_files"]["train"])
        self.assertTrue(summary["identified_files"]["test"])

    def test_detect_submission_example_alias(self):
        from src.inspect_data import inspect_raw_data

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "data" / "raw_kaggle"
            report_dir = tmp_path / "outputs" / "reports"
            working_dir = tmp_path / "working"
            data_dir = tmp_path / "data"
            raw_dir.mkdir(parents=True)

            self._write_rows(
                raw_dir / "submission_example.csv",
                ["Property", "repo_path", "severity"],
                [{"Property": "1", "repo_path": "x", "severity": "High"}],
            )
            self._write_rows(
                raw_dir / "train.csv",
                ["Property", "repo_path", "severity"],
                [{"Property": "1", "repo_path": "x", "severity": "High"}],
            )
            self._write_rows(
                raw_dir / "test.csv",
                ["repo_path"],
                [{"repo_path": "x"}],
            )

            summary = inspect_raw_data(
                raw_dir=raw_dir,
                data_dir=data_dir,
                report_dir=report_dir,
                working_dir=working_dir,
            )

        sample = summary["sample_submission"]
        self.assertIsNotNone(sample)
        self.assertTrue(sample["path"].endswith("submission_example.csv"))
        self.assertEqual(summary["identified_files"]["sample_submission"], ["submission_example.csv"])

    def _write_rows(self, path, fieldnames, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
