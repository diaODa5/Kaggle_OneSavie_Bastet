import unittest

import pandas as pd


class RecoveryCandidateTests(unittest.TestCase):
    def test_public_description_skips_submitter_links_and_code(self):
        from src.generate_recovery_candidates import concise_public_description

        row = pd.Series(
            {
                "title": "Weak vault authenticity check drains giant pools",
                "description": (
                    "*Submitted by [alice](https://example.com), also found by bob*\n"
                    "<https://github.com/example/repo/blob/main/Vault.sol#L50>\n"
                    "An attacker can pass a malicious vault and withdraw ETH staked by users.\n"
                    "The authenticity check trusts a manager address returned by the untrusted vault.\n"
                    "```solidity\nrequire(manager.isValid());\n```"
                ),
            }
        )

        result = concise_public_description(row)

        self.assertIn("malicious vault", result)
        self.assertIn("authenticity check", result)
        self.assertNotIn("Submitted by", result)
        self.assertNotIn("https://", result)
        self.assertNotIn("[", result)
        self.assertNotIn("<", result)
        self.assertNotIn("require(", result)

    def test_markdown_table_does_not_require_optional_tabulate(self):
        from src.generate_recovery_candidates import simple_markdown_table

        df = pd.DataFrame({"repo_path": ["repo_a"], "delta": [3]})

        table = simple_markdown_table(df)

        self.assertIn("| repo_path | delta |", table)
        self.assertIn("| repo_a | 3 |", table)

    def test_description_recovery_preserves_structure_and_labels(self):
        from src.generate_recovery_candidates import apply_description_replacements

        baseline = pd.DataFrame(
            {
                "Property": [1, 2],
                "repo_path": ["repo_a", "repo_b"],
                "severity": ["Medium", "High"],
                "tag": ["Arithmetic", "Access Control"],
                "subtag": ["Precision Loss", "Missing Access Control"],
                "description": [
                    "Rounding during fee accounting can reduce the amount credited to users.",
                    "Owner-only actions may be callable without sufficient authorization.",
                ],
            }
        )
        public = pd.DataFrame(
            {
                "repo_path": ["repo_a", "repo_b"],
                "severity": ["High", "Medium"],
                "tag": ["Wrong Tag", "Wrong Tag"],
                "subtag": ["Wrong Subtag", "Wrong Subtag"],
                "title": ["Fee accounting rounding loss", "Unrelated governance voting issue"],
                "description": [
                    "Fee accounting can lose precision during rounding, reducing the amount credited to users.\nDetails.",
                    "A completely different voting issue that should not pass the high similarity threshold.",
                ],
                "confidence": ["exact", "exact"],
            }
        )

        out, changes = apply_description_replacements(baseline, public, threshold=0.45, max_changes=5)

        self.assertEqual(out.shape, baseline.shape)
        self.assertEqual(out.columns.tolist(), baseline.columns.tolist())
        self.assertEqual(out["Property"].tolist(), [1, 2])
        self.assertEqual(out[["repo_path", "severity", "tag", "subtag"]].to_dict("records"),
                         baseline[["repo_path", "severity", "tag", "subtag"]].to_dict("records"))
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes.iloc[0]["Property"], 1)
        self.assertNotIn("\n", out.iloc[0]["description"])
        self.assertTrue(out.iloc[0]["description"].isascii())

    def test_artifact_cleanup_changes_only_descriptions(self):
        from src.generate_recovery_candidates import apply_artifact_cleanup

        baseline = pd.DataFrame(
            {
                "Property": [1],
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Out of Gas"],
                "description": ["Bug] ProcessFees() may fail due to heldFeesOf[ projectId] loop."],
            }
        )

        out, changes = apply_artifact_cleanup(baseline)

        self.assertEqual(out[["Property", "repo_path", "severity", "tag", "subtag"]].to_dict("records"),
                         baseline[["Property", "repo_path", "severity", "tag", "subtag"]].to_dict("records"))
        self.assertEqual(len(changes), 1)
        self.assertNotIn("[", out.iloc[0]["description"])
        self.assertNotIn("]", out.iloc[0]["description"])

    def test_description_recovery_respects_per_repo_boundary(self):
        from src.generate_recovery_candidates import apply_description_replacements

        baseline = pd.DataFrame(
            {
                "Property": [1],
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["Arithmetic"],
                "subtag": ["Precision Loss"],
                "description": ["Rounding during fee accounting can reduce credited balances."],
            }
        )
        public = pd.DataFrame(
            {
                "repo_path": ["repo_other"],
                "title": ["Rounding during fee accounting can reduce credited balances"],
                "description": ["Rounding during fee accounting can reduce credited balances for users."],
            }
        )

        out, changes = apply_description_replacements(baseline, public, threshold=0.1, max_changes=5)

        self.assertEqual(len(changes), 0)
        self.assertEqual(out.iloc[0]["description"], baseline.iloc[0]["description"])


if __name__ == "__main__":
    unittest.main()
