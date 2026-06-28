import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate_public_judging_proxy import (
    evaluate_train_proxy,
    legalize_label,
    map_public_findings,
    multiset_f1,
    repository_metrics,
    run,
)


LEGAL = {
    "severity": {"High", "Medium"},
    "tag": {"Access Control", "Arithmetic", "Reentrancy"},
    "subtag": {"Invalid Validation", "Precision Loss", "Violating CEI / Missing nonReentrant"},
}


class LabelMappingTests(unittest.TestCase):
    def test_legalize_label_accepts_only_normalized_legal_values(self):
        self.assertEqual(legalize_label(" access-control ", LEGAL["tag"]), "Access Control")
        self.assertEqual(legalize_label("HIGH", LEGAL["severity"]), "High")
        self.assertIsNone(legalize_label("Oracle", LEGAL["tag"]))

    def test_direct_raw_labels_take_precedence_over_nearest_description(self):
        train = pd.DataFrame(
            [
                {
                    "repo_path": "train-b",
                    "severity": "Medium",
                    "tag": "Arithmetic",
                    "subtag": "Precision Loss",
                    "description": "Division truncates value and causes precision loss.",
                }
            ]
        )
        findings = pd.DataFrame(
            [
                {
                    "repo_path": "public-a",
                    "severity": "High",
                    "raw_tag": "Access-Control",
                    "raw_subtag": "invalid validation",
                    "description": "Zephyr cobalt narrows the settlement path.",
                }
            ]
        )

        mapped = map_public_findings(findings, train, LEGAL, similarity_threshold=0.1)

        self.assertEqual(mapped.loc[0, "severity"], "High")
        self.assertEqual(mapped.loc[0, "raw_severity"], "High")
        self.assertEqual(mapped.loc[0, "tag"], "Access Control")
        self.assertEqual(mapped.loc[0, "subtag"], "Invalid Validation")
        self.assertEqual(mapped.loc[0, "tag_mapping_method"], "direct")
        self.assertEqual(mapped.loc[0, "subtag_mapping_method"], "direct")

    def test_blank_raw_label_falls_back_to_legal_supplied_label(self):
        findings = pd.DataFrame(
            [
                {
                    "repo_path": "public-a",
                    "severity": "Medium",
                    "tag": "Access Control",
                    "subtag": "Invalid Validation",
                    "raw_tag": "",
                    "raw_subtag": None,
                    "description": "No rule vocabulary is needed.",
                }
            ]
        )
        empty_reference = pd.DataFrame(columns=["repo_path", "severity", "tag", "subtag", "description"])

        mapped = map_public_findings(findings, empty_reference, LEGAL)

        self.assertEqual(mapped.loc[0, "tag"], "Access Control")
        self.assertEqual(mapped.loc[0, "subtag"], "Invalid Validation")
        self.assertEqual(mapped.loc[0, "tag_mapping_method"], "direct")

    def test_nearest_description_requires_threshold_and_records_similarity(self):
        train = pd.DataFrame(
            [
                {
                    "repo_path": "train-b",
                    "severity": "Medium",
                    "tag": "Arithmetic",
                    "subtag": "Precision Loss",
                    "description": "Zephyr cobalt narrows the settlement path.",
                }
            ]
        )
        finding = pd.DataFrame(
            [
                {
                    "repo_path": "public-a",
                    "severity": "Medium",
                    "description": "Zephyr cobalt narrows the settlement path.",
                }
            ]
        )

        accepted = map_public_findings(finding, train, LEGAL, similarity_threshold=0.95)
        rejected = map_public_findings(finding, train, LEGAL, similarity_threshold=1.01)

        self.assertEqual(accepted.loc[0, "tag"], "Arithmetic")
        self.assertEqual(accepted.loc[0, "tag_mapping_method"], "nearest_description")
        self.assertAlmostEqual(accepted.loc[0, "nearest_train_similarity"], 1.0)
        self.assertEqual(rejected.loc[0, "tag"], "")
        self.assertEqual(rejected.loc[0, "tag_mapping_method"], "unresolved")

    def test_deterministic_rules_return_only_legal_labels(self):
        findings = pd.DataFrame(
            [
                {
                    "repo_path": "public-a",
                    "severity": "Medium",
                    "raw_tag": "not-a-kaggle-label",
                    "description": "The external call permits reentrancy because state is updated afterwards.",
                },
                {
                    "repo_path": "public-b",
                    "severity": "Medium",
                    "raw_tag": "also-illegal",
                    "description": "A completely unmatched issue with no taxonomy terms.",
                },
            ]
        )
        empty_reference = pd.DataFrame(columns=["repo_path", "severity", "tag", "subtag", "description"])

        mapped = map_public_findings(findings, empty_reference, LEGAL)

        self.assertEqual(mapped.loc[0, "tag"], "Reentrancy")
        self.assertEqual(mapped.loc[0, "subtag"], "Violating CEI / Missing nonReentrant")
        self.assertEqual(mapped.loc[0, "tag_mapping_method"], "rule")
        self.assertEqual(mapped.loc[1, "tag"], "")
        self.assertTrue(set(mapped["tag"]) <= LEGAL["tag"] | {""})
        self.assertTrue(set(mapped["subtag"]) <= LEGAL["subtag"] | {""})


class ProxyMetricTests(unittest.TestCase):
    def test_multiset_f1_counts_duplicate_tuples(self):
        truth = [("High", "A", "X"), ("High", "A", "X"), ("Medium", "B", "Y")]
        predicted = [("High", "A", "X"), ("Medium", "B", "Y"), ("Medium", "B", "Y")]

        self.assertAlmostEqual(multiset_f1(truth, predicted), 2 / 3)
        self.assertEqual(multiset_f1([], []), 1.0)
        self.assertEqual(multiset_f1(truth, []), 0.0)

    def test_repository_metrics_include_count_error_and_tuple_multiset_f1(self):
        truth = pd.DataFrame(
            [
                {"severity": "High", "tag": "A", "subtag": "X"},
                {"severity": "High", "tag": "A", "subtag": "X"},
                {"severity": "Medium", "tag": "B", "subtag": "Y"},
            ]
        )
        predicted = pd.DataFrame(
            [
                {"severity": "High", "tag": "A", "subtag": "X"},
                {"severity": "Medium", "tag": "B", "subtag": "Y"},
            ]
        )

        metrics = repository_metrics(truth, predicted)

        self.assertEqual(metrics["true_count"], 3)
        self.assertEqual(metrics["pred_count"], 2)
        self.assertEqual(metrics["count_error"], -1)
        self.assertEqual(metrics["count_abs_error"], 1)
        self.assertAlmostEqual(metrics["tuple_multiset_f1"], 0.8)

    def test_proxy_nearest_mapping_excludes_the_evaluated_repository(self):
        train = pd.DataFrame(
            [
                {
                    "repo_path": "repo-a",
                    "severity": "High",
                    "tag": "Access Control",
                    "subtag": "Invalid Validation",
                    "description": "Unique zephyr cobalt wording.",
                },
                {
                    "repo_path": "repo-b",
                    "severity": "Medium",
                    "tag": "Arithmetic",
                    "subtag": "Precision Loss",
                    "description": "Division truncation causes numeric precision loss.",
                },
            ]
        )
        public = pd.DataFrame(
            [
                {
                    "repo_path": "repo-a",
                    "severity": "High",
                    "description": "Unique zephyr cobalt wording.",
                    "confidence": "exact",
                }
            ]
        )

        mapped, per_repo, summary = evaluate_train_proxy(
            public,
            train,
            LEGAL,
            similarity_threshold=0.9,
        )

        self.assertEqual(mapped.loc[0, "tag"], "")
        self.assertEqual(mapped.loc[0, "tag_mapping_method"], "unresolved")
        self.assertEqual(per_repo.loc[0, "pred_count"], 1)
        self.assertEqual(per_repo.loc[0, "count_abs_error"], 0)
        self.assertEqual(per_repo.loc[0, "tuple_multiset_f1"], 0.0)
        self.assertIn("exact", set(summary["verification_confidence"]))


class RunTests(unittest.TestCase):
    def test_run_writes_blocked_report_and_empty_parquet_without_upstream_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "mapped.parquet"
            report = root / "report.md"
            missing = root / "missing.parquet"
            train_path = root / "train.parquet"
            pd.DataFrame(
                columns=["repo_path", "severity", "tag", "subtag", "description"]
            ).to_parquet(train_path, index=False)

            mapped = run(
                findings_path=missing,
                train_path=train_path,
                output_path=output,
                report_path=report,
            )

            self.assertTrue(mapped.empty)
            self.assertTrue(output.exists())
            self.assertIn("Blocked", report.read_text(encoding="utf-8"))
            self.assertIn(str(missing), report.read_text(encoding="utf-8"))

    def test_run_maps_test_only_findings_but_marks_calibration_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_path = root / "findings.parquet"
            train_path = root / "train.parquet"
            output = root / "mapped.parquet"
            report = root / "report.md"
            pd.DataFrame(
                [
                    {
                        "repo_path": "test-repo",
                        "severity": "High",
                        "description": "An unauthorized caller bypasses access control.",
                        "confidence": "strong",
                    }
                ]
            ).to_parquet(findings_path, index=False)
            pd.DataFrame(
                [
                    {
                        "repo_path": "train-repo",
                        "severity": "Medium",
                        "tag": "Access Control",
                        "subtag": "Invalid Validation",
                        "description": "A role check is missing.",
                    }
                ]
            ).to_parquet(train_path, index=False)

            mapped = run(findings_path, train_path, output, report)
            report_text = report.read_text(encoding="utf-8")

            self.assertEqual(mapped.loc[0, "verification_confidence"], "strong")
            self.assertIn("Calibration unavailable", report_text)
            self.assertIn("Operational default description threshold: `0.55`", report_text)
            self.assertNotIn("Recommended description threshold", report_text)


if __name__ == "__main__":
    unittest.main()
