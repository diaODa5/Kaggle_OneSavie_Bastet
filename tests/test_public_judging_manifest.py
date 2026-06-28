import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from src.build_public_judging_manifest import (
    build_manifest,
    derive_public_judging_identity,
    read_test_commit_metadata,
    write_report,
)


class PublicJudgingManifestTests(unittest.TestCase):
    def test_sherlock_origin_maps_to_judging_repository(self):
        identity = derive_public_judging_identity(
            origin_url="https://github.com/sherlock-audit/2024-02-rubicon-finance.git",
            origin_slug="2024-02-rubicon-finance",
        )

        self.assertEqual(identity["platform"], "sherlock")
        self.assertEqual(identity["contest_slug"], "2024-02-rubicon-finance")
        self.assertEqual(
            identity["source_repo_url"],
            "https://github.com/sherlock-audit/2024-02-rubicon-finance",
        )
        self.assertEqual(
            identity["public_judging_url"],
            "https://github.com/sherlock-audit/2024-02-rubicon-finance-judging",
        )
        self.assertEqual(identity["resolution_status"], "resolved")

    def test_code4rena_origin_maps_to_findings_repository(self):
        identity = derive_public_judging_identity(
            origin_url="git@github.com:code-423n4/2024-03-coinbase.git",
            origin_slug="2024-03-coinbase",
        )

        self.assertEqual(identity["platform"], "code4rena")
        self.assertEqual(identity["contest_slug"], "2024-03-coinbase")
        self.assertEqual(
            identity["source_repo_url"],
            "https://github.com/code-423n4/2024-03-coinbase",
        )
        self.assertEqual(
            identity["public_judging_url"],
            "https://github.com/code-423n4/2024-03-coinbase-findings",
        )

    def test_missing_origin_remains_unresolved(self):
        identity = derive_public_judging_identity(origin_url="", origin_slug="")

        self.assertEqual(identity["platform"], "")
        self.assertEqual(identity["contest_slug"], "")
        self.assertEqual(identity["source_repo_url"], "")
        self.assertEqual(identity["public_judging_url"], "")
        self.assertEqual(identity["resolution_status"], "unresolved")

    def test_dependency_origin_is_not_treated_as_project_origin(self):
        identity = derive_public_judging_identity(
            origin_url="https://github.com/foundry-rs/forge-std",
            origin_slug="forge-std",
        )

        self.assertEqual(identity["platform"], "")
        self.assertEqual(identity["contest_slug"], "")
        self.assertEqual(identity["public_judging_url"], "")
        self.assertEqual(identity["resolution_status"], "unresolved")
        self.assertEqual(identity["provenance"], "ignored_dependency_origin")

    def test_official_match_fills_code4rena_identity_and_preserves_metadata(self):
        test_repos = pd.DataFrame({"repo_path": ["sherlock", "matched", "missing"]})
        identities = pd.DataFrame(
            [
                {
                    "repo_path": "sherlock",
                    "origin_url": "https://github.com/sherlock-audit/2023-12-dodo.git",
                    "origin_slug": "2023-12-dodo",
                },
                {"repo_path": "missing", "origin_url": "", "origin_slug": ""},
            ]
        )
        official_matches = pd.DataFrame(
            [
                {
                    "split": "test",
                    "repo_path": "matched",
                    "matched_external_repo_path": "repos/2024-02-thruster",
                    "confidence": "very_high",
                    "score": 317.0,
                    "matched_fingerprints": "solidity_content_aggregate_sha256",
                }
            ]
        )
        commits = {
            "sherlock": {
                "test_git_head": "ref: refs/heads/main",
                "test_commit": "a" * 40,
                "test_commit_provenance": ".git/refs/heads/main",
            }
        }

        manifest = build_manifest(test_repos, identities, official_matches, commits)

        self.assertEqual(list(manifest["repo_path"]), ["matched", "missing", "sherlock"])
        self.assertEqual(len(manifest), 3)
        matched = manifest.set_index("repo_path").loc["matched"]
        self.assertEqual(matched["platform"], "code4rena")
        self.assertEqual(matched["contest_slug"], "2024-02-thruster")
        self.assertEqual(matched["provenance"], "official_fingerprint_match")
        self.assertTrue(matched["public_archive_verified"])
        self.assertEqual(matched["match_confidence"], "very_high")
        self.assertEqual(matched["match_score"], 317.0)
        missing = manifest.set_index("repo_path").loc["missing"]
        self.assertEqual(missing["resolution_status"], "unresolved")
        sherlock = manifest.set_index("repo_path").loc["sherlock"]
        self.assertEqual(sherlock["provenance"], "test_git_origin")
        self.assertTrue(sherlock["public_archive_verified"])
        self.assertEqual(sherlock["test_git_head"], "ref: refs/heads/main")
        self.assertEqual(sherlock["test_commit"], "a" * 40)
        self.assertEqual(sherlock["test_commit_provenance"], ".git/refs/heads/main")

    def test_read_test_commit_metadata_resolves_symbolic_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("test/repo-a/.git/HEAD", "ref: refs/heads/main\n")
                zf.writestr("test/repo-a/.git/refs/heads/main", f"{'b' * 40}\n")
                zf.writestr("test/repo-b/contracts/Example.sol", "contract Example {}")

            metadata = read_test_commit_metadata(zip_path)

        self.assertEqual(
            metadata["repo-a"],
            {
                "test_git_head": "ref: refs/heads/main",
                "test_commit": "b" * 40,
                "test_commit_provenance": ".git/refs/heads/main",
            },
        )
        self.assertNotIn("repo-b", metadata)

    def test_report_uses_stable_project_relative_output_path(self):
        manifest = build_manifest(
            pd.DataFrame({"repo_path": ["missing"]}),
            pd.DataFrame(columns=["repo_path", "origin_url", "origin_slug"]),
            pd.DataFrame(columns=["split", "repo_path", "matched_external_repo_path"]),
            {},
        )
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            write_report(manifest, report_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("- output: `data/processed/public_judging_manifest.csv`", report)


if __name__ == "__main__":
    unittest.main()
