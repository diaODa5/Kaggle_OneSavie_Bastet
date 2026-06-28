from __future__ import annotations

import unittest

import pandas as pd

from src.generate_confidence_pruned_395 import (
    DROP_DECISIONS,
    TARGETED_V2_PROPERTIES,
    build_confidence_pruned_submission,
)


class ConfidencePruned395Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.read_csv("outputs/submission_backup_3825_no_padding_targeted_v2.csv")

    def test_builds_395_findings_with_five_padding_rows(self) -> None:
        result, dropped = build_confidence_pruned_submission(self.base)

        padding = result["repo_path"].astype(str).str.lower().eq("empty")
        self.assertEqual(len(result), 400)
        self.assertEqual(int((~padding).sum()), 395)
        self.assertEqual(int(padding.sum()), 5)
        self.assertEqual(result["Property"].tolist(), list(range(1, 401)))
        self.assertEqual(set(dropped["Property"]), set(DROP_DECISIONS))

    def test_preserves_all_targeted_v2_findings(self) -> None:
        result, _ = build_confidence_pruned_submission(self.base)

        baseline_targeted = self.base[self.base["Property"].isin(TARGETED_V2_PROPERTIES)]
        result_keys = set(
            result.loc[result["repo_path"].ne("empty"), ["repo_path", "severity", "tag", "subtag", "description"]]
            .astype(str)
            .agg("\x1f".join, axis=1)
        )
        targeted_keys = set(
            baseline_targeted[["repo_path", "severity", "tag", "subtag", "description"]]
            .astype(str)
            .agg("\x1f".join, axis=1)
        )
        self.assertLessEqual(targeted_keys, result_keys)

    def test_padding_uses_official_empty_sentinel_in_every_field(self) -> None:
        result, _ = build_confidence_pruned_submission(self.base)

        padding = result[result["repo_path"].eq("empty")]
        self.assertTrue(
            padding[["repo_path", "severity", "tag", "subtag", "description"]]
            .eq("empty")
            .all(axis=None)
        )


if __name__ == "__main__":
    unittest.main()
