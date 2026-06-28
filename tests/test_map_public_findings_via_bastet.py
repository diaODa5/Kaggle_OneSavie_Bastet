import unittest
import tempfile
from pathlib import Path

import pandas as pd


def training_rows(include_combined=True):
    rows = [
        {
            "severity": "High",
            "tag": "Access Control",
            "subtag": "Invalid Validation",
            "description": "Authorization is not checked before withdrawal.",
        },
        {
            "severity": "High",
            "tag": "Reentrancy",
            "subtag": "Violating CEI / Missing nonReentrant",
            "description": "An external call happens before state is updated.",
        },
        {
            "severity": "Medium",
            "tag": "Arithmetic",
            "subtag": "Precision Loss",
            "description": "Division truncation loses precision.",
        },
    ]
    if include_combined:
        rows.append(
            {
                "severity": "High",
                "tag": "Access Control, Reentrancy",
                "subtag": (
                    "Invalid Validation, "
                    "Violating CEI / Missing nonReentrant"
                ),
                "description": "Authorization and callback ordering are both unsafe.",
            }
        )
    return pd.DataFrame(rows)


def public_finding(**overrides):
    row = {
        "repo_path": "public-repo",
        "platform": "code4rena",
        "contest": "2024-01-demo-findings",
        "issue_id": "M-01",
        "title": "Withdrawal callback bypasses authorization",
        "severity": "High",
        "description": "The callback runs before authorization and state updates.",
    }
    row.update(overrides)
    return row


def external_finding(**overrides):
    row = {
        "repo_path": "repos/2024-01-demo",
        "report_path": "reports/2024-01-demo/m_1.md",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Invalid Validation",
        "description": "The callback runs before authorization and state updates.",
        "report_text": "The callback runs before authorization and state updates.",
        "source_csv": "C:/bastet/dataset.csv",
        "tag_mapping_method": "rule_text",
        "subtag_mapping_method": "rule_text",
    }
    row.update(overrides)
    return row


class NormalizationTests(unittest.TestCase):
    def test_normalizes_contest_and_high_medium_issue_ids(self):
        from src.map_public_findings_via_bastet import (
            normalize_contest,
            normalize_issue_id,
        )

        self.assertEqual(
            normalize_contest("repos/2024-01-demo-findings-main"),
            "2024-01-demo",
        )
        self.assertEqual(
            normalize_contest("https://github.com/code-423n4/2024-01-demo-judging.git"),
            "2024-01-demo",
        )
        self.assertEqual(normalize_issue_id("M-01"), "M-1")
        self.assertEqual(normalize_issue_id("m_1.md"), "M-1")
        self.assertEqual(normalize_issue_id("H 002"), "H-2")
        self.assertEqual(normalize_issue_id(None), "")


class ExternalVersionTests(unittest.TestCase):
    def test_duplicate_versions_merge_only_to_an_actual_train_enumeration(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame([public_finding()])
        external = pd.DataFrame(
            [
                external_finding(),
                external_finding(),
                external_finding(
                    tag="Reentrancy",
                    subtag="Violating CEI / Missing nonReentrant",
                    source_csv="C:/bastet/dataset_0831.csv",
                    tag_mapping_method="direct",
                    subtag_mapping_method="direct",
                ),
            ]
        )

        mapped = map_public_findings(findings, external, training_rows())

        self.assertEqual(mapped.loc[0, "tag"], "Access Control, Reentrancy")
        self.assertEqual(
            mapped.loc[0, "subtag"],
            "Invalid Validation, Violating CEI / Missing nonReentrant",
        )
        self.assertEqual(mapped.loc[0, "mapping_method"], "external_exact_merged")
        self.assertFalse(bool(mapped.loc[0, "mapping_ambiguity"]))
        self.assertEqual(mapped.loc[0, "mapping_candidate_count"], 2)
        self.assertTrue(bool(mapped.loc[0, "labels_complete"]))

    def test_unmergeable_versions_use_source_priority_and_record_ambiguity(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame([public_finding()])
        external = pd.DataFrame(
            [
                external_finding(),
                external_finding(
                    tag="Reentrancy",
                    subtag="Violating CEI / Missing nonReentrant",
                    source_csv="C:/bastet/dataset_0831.csv",
                    tag_mapping_method="direct",
                    subtag_mapping_method="direct",
                ),
            ]
        )

        mapped = map_public_findings(
            findings,
            external,
            training_rows(include_combined=False),
        )

        self.assertEqual(mapped.loc[0, "tag"], "Reentrancy")
        self.assertEqual(
            mapped.loc[0, "subtag"],
            "Violating CEI / Missing nonReentrant",
        )
        self.assertEqual(
            mapped.loc[0, "mapping_method"],
            "external_exact_ambiguous",
        )
        self.assertTrue(bool(mapped.loc[0, "mapping_ambiguity"]))
        self.assertGreater(float(mapped.loc[0, "mapping_confidence"]), 0.8)
        self.assertLess(float(mapped.loc[0, "mapping_confidence"]), 1.0)


class SherlockFallbackTests(unittest.TestCase):
    def test_same_issue_id_in_another_contest_is_never_used(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame(
            [
                public_finding(
                    platform="sherlock",
                    contest="sherlock-alpha",
                    issue_id="M-1",
                    title="Cross contest sentinel",
                    description="Unique cross contest sentinel wording.",
                )
            ]
        )
        external = pd.DataFrame(
            [
                external_finding(
                    repo_path="repos/sherlock-beta",
                    report_path="reports/sherlock-beta/m_1.md",
                    description="Unique cross contest sentinel wording.",
                    report_text="Unique cross contest sentinel wording.",
                )
            ]
        )

        mapped = map_public_findings(findings, external, training_rows())

        self.assertEqual(mapped.loc[0, "tag"], "")
        self.assertEqual(mapped.loc[0, "subtag"], "")
        self.assertEqual(mapped.loc[0, "mapping_method"], "unresolved")
        self.assertEqual(mapped.loc[0, "mapping_candidate_count"], 0)
        self.assertFalse(bool(mapped.loc[0, "labels_complete"]))

    def test_existing_legal_labels_are_preserved_before_fallback(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame(
            [
                public_finding(
                    platform="sherlock",
                    contest="sherlock-alpha",
                    issue_id="H-1",
                    tag="access-control",
                    subtag="invalid validation",
                    description="Unrelated wording.",
                )
            ]
        )

        mapped = map_public_findings(
            findings,
            pd.DataFrame(),
            training_rows(),
        )

        self.assertEqual(mapped.loc[0, "tag"], "Access Control")
        self.assertEqual(mapped.loc[0, "subtag"], "Invalid Validation")
        self.assertEqual(mapped.loc[0, "mapping_method"], "existing_legal")
        self.assertEqual(mapped.loc[0, "mapping_confidence"], 1.0)
        self.assertTrue(bool(mapped.loc[0, "labels_complete"]))

    def test_high_similarity_train_fallback_records_method_and_confidence(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame(
            [
                public_finding(
                    platform="sherlock",
                    contest="sherlock-alpha",
                    issue_id="M-2",
                    title="",
                    severity="Medium",
                    description="Division truncation loses precision.",
                )
            ]
        )

        mapped = map_public_findings(
            findings,
            pd.DataFrame(),
            training_rows(),
            fallback_threshold=0.90,
        )

        self.assertEqual(mapped.loc[0, "tag"], "Arithmetic")
        self.assertEqual(mapped.loc[0, "subtag"], "Precision Loss")
        self.assertEqual(mapped.loc[0, "mapping_method"], "train_text_fallback")
        self.assertGreaterEqual(mapped.loc[0, "mapping_confidence"], 0.90)
        self.assertTrue(bool(mapped.loc[0, "labels_complete"]))

    def test_external_text_fallback_is_limited_to_the_same_contest(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame(
            [
                public_finding(
                    platform="sherlock",
                    contest="sherlock-alpha",
                    issue_id="M-9",
                    title="",
                    description="A callback happens before state is updated.",
                )
            ]
        )
        external = pd.DataFrame(
            [
                external_finding(
                    repo_path="repos/sherlock-alpha",
                    report_path="reports/sherlock-alpha/h_1.md",
                    tag="Reentrancy",
                    subtag="Violating CEI / Missing nonReentrant",
                    description="A callback happens before state is updated.",
                    report_text="A callback happens before state is updated.",
                )
            ]
        )

        mapped = map_public_findings(
            findings,
            external,
            training_rows(),
            fallback_threshold=0.90,
        )

        self.assertEqual(mapped.loc[0, "tag"], "Reentrancy")
        self.assertEqual(
            mapped.loc[0, "subtag"],
            "Violating CEI / Missing nonReentrant",
        )
        self.assertEqual(
            mapped.loc[0, "mapping_method"],
            "external_text_same_contest",
        )
        self.assertGreaterEqual(mapped.loc[0, "mapping_confidence"], 0.90)


class LegalityAndDescriptionTests(unittest.TestCase):
    def test_null_and_illegal_labels_are_empty_and_public_root_cause_wins(self):
        from src.map_public_findings_via_bastet import map_public_findings

        findings = pd.DataFrame(
            [
                public_finding(
                    platform="sherlock",
                    contest=None,
                    issue_id=None,
                    title="Final public title",
                    root_cause="Specific public root cause.",
                    description="A longer public body.",
                    severity=None,
                    tag="invented label",
                    subtag=None,
                )
            ]
        )

        mapped = map_public_findings(
            findings,
            pd.DataFrame(),
            training_rows(),
        )

        self.assertEqual(mapped.loc[0, "description"], "Final public title\n\nSpecific public root cause.")
        self.assertEqual(mapped.loc[0, "severity"], "")
        self.assertEqual(mapped.loc[0, "tag"], "")
        self.assertEqual(mapped.loc[0, "subtag"], "")
        self.assertFalse(bool(mapped.loc[0, "labels_complete"]))
        for column in ("severity", "tag", "subtag"):
            legal = set(training_rows()[column])
            self.assertIn(mapped.loc[0, column], legal | {""})


class RunTests(unittest.TestCase):
    def test_run_writes_mapped_parquet_and_required_report_counts(self):
        from src.map_public_findings_via_bastet import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_path = root / "findings.parquet"
            external_path = root / "external.parquet"
            train_path = root / "train.csv"
            output_path = root / "mapped.parquet"
            report_path = root / "report.md"

            pd.DataFrame(
                [
                    public_finding(),
                    public_finding(
                        issue_id="M-99",
                        title="No exact candidate",
                        description="No matching taxonomy text.",
                    ),
                    public_finding(
                        platform="sherlock",
                        contest="sherlock-alpha",
                        issue_id="M-2",
                        title="",
                        severity="Medium",
                        description="Division truncation loses precision.",
                    ),
                ]
            ).to_parquet(findings_path, index=False)
            pd.DataFrame(
                [
                    external_finding(),
                    external_finding(
                        tag="Reentrancy",
                        subtag="Violating CEI / Missing nonReentrant",
                        source_csv="C:/bastet/dataset_0831.csv",
                        tag_mapping_method="direct",
                        subtag_mapping_method="direct",
                    ),
                ]
            ).to_parquet(external_path, index=False)
            training_rows(include_combined=False).to_csv(train_path, index=False)

            mapped = run(
                findings_path=findings_path,
                external_path=external_path,
                train_path=train_path,
                output_path=output_path,
                report_path=report_path,
                fallback_threshold=0.90,
            )

            saved = pd.read_parquet(output_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(len(mapped), 3)
        self.assertEqual(len(saved), 3)
        self.assertIn("Exact contest + issue mappings: `1`", report)
        self.assertIn("Ambiguous mappings: `1`", report)
        self.assertIn("Sherlock fallbacks: `1`", report)
        self.assertIn("Unresolved findings: `1`", report)
        self.assertTrue(
            {
                "severity",
                "tag",
                "subtag",
                "mapping_method",
                "mapping_confidence",
                "labels_complete",
            }.issubset(saved.columns)
        )


if __name__ == "__main__":
    unittest.main()
