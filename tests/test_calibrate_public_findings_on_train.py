import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd


def git_config(url: str) -> str:
    return (
        '[core]\n'
        '\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        f"\turl = {url}\n"
        "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
    )


class TrainOriginMappingTests(unittest.TestCase):
    def test_dependency_origin_is_excluded_and_nested_dependency_config_is_ignored(self):
        from src.calibrate_public_findings_on_train import (
            parse_main_project_origin,
            read_train_origin_evidence,
        )

        ignored = parse_main_project_origin(
            git_config("https://github.com/foundry-rs/forge-std")
        )
        self.assertEqual(ignored["external_repo_path"], "")
        self.assertEqual(ignored["rejection_reason"], "dependency_origin")

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "train.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(
                    "train/repo-a/.git/config",
                    git_config("https://github.com/code-423n4/2024-01-main.git"),
                )
                zf.writestr(
                    "train/repo-a/lib/forge-std/.git/config",
                    git_config("https://github.com/foundry-rs/forge-std"),
                )
                zf.writestr(
                    "train/repo-b/.git/config",
                    git_config(
                        "https://github.com/OpenZeppelin/openzeppelin-contracts.git"
                    ),
                )

            evidence = read_train_origin_evidence(archive, {"repo-a", "repo-b"})

        by_repo = evidence.set_index("train_repo_path")
        self.assertEqual(
            by_repo.loc["repo-a", "external_repo_path"],
            "repos/2024-01-main",
        )
        self.assertEqual(by_repo.loc["repo-a", "mapping_source"], "git_origin")
        self.assertEqual(by_repo.loc["repo-b", "external_repo_path"], "")
        self.assertEqual(
            by_repo.loc["repo-b", "rejection_reason"],
            "dependency_origin",
        )

    def test_repo_mapping_uses_origin_then_train_pair_then_train_fingerprint(self):
        from src.calibrate_public_findings_on_train import resolve_train_repo_mapping

        origin = pd.DataFrame(
            [
                {
                    "train_repo_path": "origin-repo",
                    "origin_url": "https://github.com/code-423n4/origin-project.git",
                    "contest": "origin-project",
                    "external_repo_path": "repos/origin-project",
                    "mapping_source": "git_origin",
                    "mapping_confidence": "exact",
                    "rejection_reason": "",
                }
            ]
        )
        train_pairs = pd.DataFrame(
            [
                {
                    "train_repo_path": "origin-repo",
                    "top_external_repo_path": "repos/wrong-pair",
                    "confidence": "very_high",
                },
                {
                    "train_repo_path": "pair-repo",
                    "top_external_repo_path": "repos/pair-project",
                    "confidence": "high",
                },
            ]
        )
        official = pd.DataFrame(
            [
                {
                    "split": "test",
                    "repo_path": "fingerprint-repo",
                    "matched_external_repo_path": "repos/test-leak",
                    "confidence": "very_high",
                    "score": 999.0,
                },
                {
                    "split": "train",
                    "repo_path": "fingerprint-repo",
                    "matched_external_repo_path": "repos/fingerprint-project",
                    "confidence": "very_high",
                    "score": 317.0,
                },
                {
                    "split": "train",
                    "repo_path": "low-repo",
                    "matched_external_repo_path": "repos/low-confidence",
                    "confidence": "low",
                    "score": 10.0,
                },
            ]
        )

        mapping = resolve_train_repo_mapping(
            ["origin-repo", "pair-repo", "fingerprint-repo", "low-repo"],
            origin,
            train_pairs,
            official,
        ).set_index("train_repo_path")

        self.assertEqual(
            mapping.loc["origin-repo", "external_repo_path"],
            "repos/origin-project",
        )
        self.assertEqual(mapping.loc["origin-repo", "mapping_source"], "git_origin")
        self.assertEqual(
            mapping.loc["pair-repo", "external_repo_path"],
            "repos/pair-project",
        )
        self.assertEqual(
            mapping.loc["pair-repo", "mapping_source"],
            "high_confidence_train_pair",
        )
        self.assertEqual(
            mapping.loc["fingerprint-repo", "external_repo_path"],
            "repos/fingerprint-project",
        )
        self.assertEqual(
            mapping.loc["fingerprint-repo", "mapping_source"],
            "official_train_fingerprint",
        )
        self.assertEqual(mapping.loc["low-repo", "external_repo_path"], "")
        self.assertEqual(mapping.loc["low-repo", "mapping_source"], "unresolved")


class CandidateMatchingTests(unittest.TestCase):
    def test_description_similarity_is_clipped_to_probability_bounds(self):
        from src.calibrate_public_findings_on_train import (
            greedy_one_to_one_matches,
        )

        description = (
            "If the contract call on the destination chain fails and the contract "
            "on the source chain lacks a mechanism to return the burned tokens or "
            "re-mint them, it may result in a loss of user assets."
        )
        truth = pd.DataFrame([{"description": description}])
        candidates = pd.DataFrame([{"description": description}])

        matches = greedy_one_to_one_matches(truth, candidates)

        self.assertTrue(matches["description_similarity"].between(0.0, 1.0).all())

    def test_greedy_matching_is_one_to_one(self):
        from src.calibrate_public_findings_on_train import (
            greedy_one_to_one_matches,
        )

        truth = pd.DataFrame(
            [
                {
                    "severity": "High",
                    "tag": "Access Control",
                    "subtag": "Missing Check",
                    "description": "Unauthorized users can withdraw every vault asset.",
                },
                {
                    "severity": "Medium",
                    "tag": "Oracle",
                    "subtag": "Stale Price",
                    "description": "The oracle accepts a stale price during settlement.",
                },
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "severity": "Medium",
                    "tag": "Oracle",
                    "subtag": "Stale Price",
                    "description": "The oracle accepts a stale price during settlement.",
                },
                {
                    "severity": "High",
                    "tag": "Access Control",
                    "subtag": "Missing Check",
                    "description": "Unauthorized users can withdraw every vault asset.",
                },
            ]
        )

        matches = greedy_one_to_one_matches(truth, candidates)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches["truth_index"].nunique(), 2)
        self.assertEqual(matches["candidate_index"].nunique(), 2)
        self.assertEqual(
            set(zip(matches["truth_index"], matches["candidate_index"])),
            {(0, 1), (1, 0)},
        )
        self.assertTrue((matches["description_similarity"] > 0.99).all())

    def test_multilabel_values_are_order_insensitive_and_duplicate_candidates_are_removed(self):
        from src.calibrate_public_findings_on_train import (
            deduplicate_candidates,
            repository_metrics,
        )

        truth = pd.DataFrame(
            [
                {
                    "severity": "High",
                    "tag": "Access Control, Reentrancy",
                    "subtag": "Missing Check, External Call",
                    "description": "The callback occurs before authorization is checked.",
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "severity": "high",
                    "tag": "Reentrancy, Access Control",
                    "subtag": "External Call, Missing Check",
                    "description": "The callback occurs before authorization is checked.",
                    "report_path": "reports/example/h_1.md",
                },
                {
                    "severity": "High",
                    "tag": "Access Control, Reentrancy",
                    "subtag": "Missing Check, External Call",
                    "description": "  The callback occurs before authorization is checked. ",
                    "report_path": "reports/example/duplicate.md",
                },
            ]
        )

        deduplicated = deduplicate_candidates(candidates)
        metrics = repository_metrics(truth, deduplicated)

        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(metrics["severity_multiset_f1"], 1.0)
        self.assertEqual(metrics["tag_multiset_f1"], 1.0)
        self.assertEqual(metrics["subtag_multiset_f1"], 1.0)
        self.assertEqual(metrics["tuple_multiset_f1"], 1.0)


class CalibrationRunTests(unittest.TestCase):
    def test_run_writes_per_repo_calibration_and_leakage_safe_report(self):
        from src.calibrate_public_findings_on_train import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            train_zip = root / "train.zip"
            external_path = root / "external.parquet"
            train_pairs = root / "train_pairs.csv"
            official_matches = root / "official_matches.parquet"
            output_path = root / "calibration.parquet"
            report_path = root / "report.md"

            pd.DataFrame(
                [
                    {
                        "Property": 1,
                        "repo_path": "repo-a",
                        "severity": "High",
                        "tag": "Access Control",
                        "subtag": "Missing Check",
                        "description": "Unauthorized users can withdraw every vault asset.",
                    },
                    {
                        "Property": 2,
                        "repo_path": "repo-a",
                        "severity": "Medium",
                        "tag": "Oracle",
                        "subtag": "Stale Price",
                        "description": "The oracle accepts a stale price during settlement.",
                    },
                ]
            ).to_csv(train_csv, index=False)
            with zipfile.ZipFile(train_zip, "w") as zf:
                zf.writestr(
                    "train/repo-a/.git/config",
                    git_config("https://github.com/code-423n4/2024-01-main.git"),
                )
            pd.DataFrame(
                [
                    {
                        "repo_path": "repos/2024-01-main",
                        "severity": "High",
                        "tag": "Access Control",
                        "subtag": "Missing Check",
                        "description": "Unauthorized users can withdraw every vault asset.",
                        "report_path": "reports/2024-01-main/h_1.md",
                    },
                    {
                        "repo_path": "repos/2024-01-main",
                        "severity": "Medium",
                        "tag": "Oracle",
                        "subtag": "Stale Price",
                        "description": "The oracle accepts a stale price during settlement.",
                        "report_path": "reports/2024-01-main/m_1.md",
                    },
                ]
            ).to_parquet(external_path, index=False)
            pd.DataFrame(
                columns=[
                    "train_repo_path",
                    "top_external_repo_path",
                    "confidence",
                ]
            ).to_csv(train_pairs, index=False)
            pd.DataFrame(
                columns=[
                    "split",
                    "repo_path",
                    "matched_external_repo_path",
                    "confidence",
                    "score",
                ]
            ).to_parquet(official_matches, index=False)

            with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as read_csv:
                result = run(
                    train_zip_path=train_zip,
                    train_csv_path=train_csv,
                    external_findings_path=external_path,
                    high_confidence_pairs_path=train_pairs,
                    official_matches_path=official_matches,
                    output_path=output_path,
                    report_path=report_path,
                )

            csv_reads = [str(call.args[0]) for call in read_csv.call_args_list]
            self.assertEqual(set(csv_reads), {str(train_csv), str(train_pairs)})
            self.assertTrue(output_path.exists())
            saved = pd.read_parquet(output_path)
            self.assertEqual(len(saved), len(result))
            self.assertEqual(
                set(saved["strategy"]),
                {
                    "all",
                    "description_exactish_0.70",
                    "description_exactish_0.85",
                    "high_only",
                    "medium_only",
                },
            )
            all_row = saved[saved["strategy"] == "all"].iloc[0]
            self.assertEqual(all_row["true_count"], 2)
            self.assertEqual(all_row["public_candidate_count"], 2)
            self.assertEqual(all_row["selected_candidate_count"], 2)
            self.assertEqual(all_row["count_error"], 0)
            self.assertGreater(all_row["top_description_similarity"], 0.99)
            self.assertEqual(all_row["tuple_multiset_f1"], 1.0)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Calibration split: `train` only", report)
            self.assertIn("Test labels read: `no`", report)
            self.assertIn("Recommended precision strategy", report)
            self.assertIn("description_exactish_0.85", report)
            self.assertIn("high_only", report)
            self.assertIn("medium_only", report)
            self.assertIn("git_origin", report)


if __name__ == "__main__":
    unittest.main()
