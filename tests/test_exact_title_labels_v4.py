from __future__ import annotations

import unittest

import pandas as pd

from src.generate_exact_title_labels_v4 import LABEL_UPDATES, build_candidate


class ExactTitleLabelsV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.read_csv("outputs/submission_exact_title_semantic_labels_v3.csv")

    def test_changes_only_configured_tag_and_subtag_fields(self) -> None:
        candidate, changes = build_candidate()

        self.assertEqual(len(candidate), 400)
        self.assertEqual(set(change["Property"] for change in changes), set(LABEL_UPDATES))
        self.assertEqual(candidate["Property"].tolist(), list(range(1, 401)))

        protected = ["Property", "repo_path", "severity", "description"]
        pd.testing.assert_frame_equal(candidate[protected], self.base[protected])
        self.assertEqual(
            candidate["repo_path"].value_counts().to_dict(),
            self.base["repo_path"].value_counts().to_dict(),
        )

        changed_mask = candidate[["tag", "subtag"]].ne(
            self.base[["tag", "subtag"]]
        ).any(axis=1)
        self.assertEqual(
            set(candidate.loc[changed_mask, "Property"].astype(int)),
            set(LABEL_UPDATES),
        )

    def test_updates_remain_exact_unambiguous_title_matches(self) -> None:
        candidate, changes = build_candidate()

        for change in changes:
            self.assertEqual(change["confidence"], "exact")
            self.assertFalse(change["mapping_ambiguity"])
            self.assertTrue(change["title_contained"])
            self.assertIn(
                change["mapping_method"],
                {"external_exact", "external_exact_merged"},
            )
            row = candidate.loc[
                candidate["Property"].astype(int).eq(int(change["Property"]))
            ].iloc[0]
            self.assertEqual(row["tag"], change["new_tag"])
            self.assertEqual(row["subtag"], change["new_subtag"])


if __name__ == "__main__":
    unittest.main()
