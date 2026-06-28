from __future__ import annotations

import unittest

import pandas as pd

from src.generate_exact_title_semantic_labels_v3 import (
    LABEL_UPDATES,
    build_candidate,
)


class ExactTitleSemanticLabelsV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.read_csv("outputs/submission_manual_label_realignment_v2_bold.csv")

    def test_changes_only_configured_tag_and_subtag_fields(self) -> None:
        candidate, changes = build_candidate()

        self.assertEqual(len(candidate), 400)
        self.assertEqual(set(change["Property"] for change in changes), set(LABEL_UPDATES))
        self.assertEqual(candidate["Property"].tolist(), list(range(1, 401)))

        protected_columns = ["Property", "repo_path", "severity", "description"]
        pd.testing.assert_frame_equal(
            candidate[protected_columns],
            self.base[protected_columns],
        )
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

    def test_every_update_uses_unique_exact_unambiguous_title_match(self) -> None:
        candidate, changes = build_candidate()
        by_property = {int(change["Property"]): change for change in changes}

        for prop, expected_key in LABEL_UPDATES.items():
            change = by_property[prop]
            self.assertEqual(change["finding_key"], expected_key)
            self.assertEqual(change["confidence"], "exact")
            self.assertFalse(change["mapping_ambiguity"])
            self.assertIn(
                change["mapping_method"],
                {"external_exact", "external_exact_merged"},
            )
            self.assertTrue(change["title_contained"])

            row = candidate.loc[candidate["Property"].astype(int).eq(prop)].iloc[0]
            self.assertEqual(row["tag"], change["new_tag"])
            self.assertEqual(row["subtag"], change["new_subtag"])


if __name__ == "__main__":
    unittest.main()
