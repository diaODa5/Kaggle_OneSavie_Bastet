import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


class SourceAuditCandidateTests(unittest.TestCase):
    def test_verify_condition_requires_all_tokens_and_no_forbidden_token(self):
        from src.generate_source_audit_candidate import verify_condition

        text = "feeAmount = amount.mulDivUp(fee, DENOM); maxFee = amount.mulDivDown(MAX_FEE, DENOM);"
        condition = {
            "required_all": ["mulDivUp", "mulDivDown", "MAX_FEE"],
            "forbidden_any": ["mulDivUp(MAX_FEE"],
        }

        passed, details = verify_condition(text, condition)

        self.assertTrue(passed)
        self.assertEqual(details["missing"], [])
        self.assertEqual(details["forbidden_present"], [])

    def test_verify_condition_rejects_present_mitigation(self):
        from src.generate_source_audit_candidate import verify_condition

        text = "feeOutput.amount += feeAmount; result[j] = feeOutput;"
        condition = {
            "required_all": ["feeOutput.amount += feeAmount"],
            "forbidden_any": ["result[j] = feeOutput"],
        }

        passed, details = verify_condition(text, condition)

        self.assertFalse(passed)
        self.assertEqual(details["forbidden_present"], ["result[j] = feeOutput"])

    def test_verify_source_requirements_do_not_combine_separate_excerpts(self):
        from src.generate_source_audit_candidate import verify_source_requirements

        excerpts = ["function first() { vulnerableCall(); }", "function second() { missingGuard(); }"]
        condition = {
            "source_required": [
                ["vulnerableCall", "missingGuard"],
                [],
            ],
            "source_forbidden": [[], []],
            "file_forbidden": [[], []],
        }

        passed, details = verify_source_requirements(excerpts, excerpts, condition)

        self.assertFalse(passed)
        self.assertIn("missingGuard", details[0]["missing"])

    def test_read_source_excerpt_rejects_out_of_bounds_lines(self):
        from src.generate_source_audit_candidate import read_source_excerpt

        with TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "sources.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("test/repo/src/Test.sol", "line1\nline2\n")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "Invalid source range"):
                    read_source_excerpt(
                        archive,
                        "repo",
                        {"path": "src/Test.sol", "start": 1, "end": 5},
                    )

    def test_apply_replacements_preserves_shape_repo_counts_and_properties(self):
        from src.generate_source_audit_candidate import apply_replacements

        baseline = pd.DataFrame(
            {
                "Property": [1, 2, 3],
                "repo_path": ["repo_a", "repo_a", "repo_b"],
                "severity": ["Medium", "High", "Medium"],
                "tag": ["DoS", "Access Control", "Arithmetic"],
                "subtag": ["Bad Condition", "Asset Theft", "Precision Loss"],
                "description": ["old a1", "old a2", "old b1"],
            }
        )
        evidence = pd.DataFrame(
            [
                {
                    "replace_property": 1,
                    "repo_path": "repo_a",
                    "severity": "Medium",
                    "tag": "Arithmetic, DoS",
                    "subtag": "Rounding Error",
                    "description": "A concrete source-proven rounding mismatch causes valid fee execution to revert.",
                    "grade": "B",
                    "verified": True,
                }
            ]
        )

        out, changes = apply_replacements(baseline, evidence, max_changes=8)

        self.assertEqual(out.shape, baseline.shape)
        self.assertEqual(out["Property"].tolist(), [1, 2, 3])
        self.assertEqual(out["repo_path"].value_counts().to_dict(), baseline["repo_path"].value_counts().to_dict())
        self.assertEqual(len(changes), 1)
        self.assertEqual(out.loc[out["Property"] == 1, "tag"].item(), "Arithmetic, DoS")

    def test_apply_replacements_rejects_cross_repo_change(self):
        from src.generate_source_audit_candidate import apply_replacements

        baseline = pd.DataFrame(
            {
                "Property": [1],
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Bad Condition"],
                "description": ["old"],
            }
        )
        evidence = pd.DataFrame(
            [
                {
                    "replace_property": 1,
                    "repo_path": "repo_b",
                    "severity": "Medium",
                    "tag": "DoS",
                    "subtag": "Bad Condition",
                    "description": "This description is long enough to be a valid finding.",
                    "grade": "B",
                    "verified": True,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "same repository"):
            apply_replacements(baseline, evidence, max_changes=8)

    def test_validate_final_candidate_rejects_non_400_rows(self):
        from src.generate_source_audit_candidate import validate_final_candidate

        sample = pd.DataFrame(
            columns=["Property", "repo_path", "severity", "tag", "subtag", "description"]
        )
        train = pd.DataFrame(
            {
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Bad Condition"],
            }
        )
        test = pd.DataFrame({"repo_path": ["repo_a"]})
        candidate = pd.DataFrame(
            {
                "Property": [1],
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Bad Condition"],
                "description": ["A sufficiently specific vulnerability description for validation."],
            }
        )

        with self.assertRaisesRegex(ValueError, "exactly 400"):
            validate_final_candidate(candidate, sample, test, train)


if __name__ == "__main__":
    unittest.main()
