from __future__ import annotations

import unittest

import pandas as pd

from src.generate_description_quality_repair_v9 import (
    DESCRIPTION_REPAIRS,
    build_description_quality_repair_v9,
)


class DescriptionQualityRepairV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.read_csv("outputs/submission_backup_40075_template_description_repair_v7.csv")

    def test_repairs_only_configured_descriptions(self) -> None:
        result, changed = build_description_quality_repair_v9(self.base)

        self.assertEqual(len(result), 400)
        self.assertFalse(result["repo_path"].astype(str).str.lower().eq("empty").any())
        self.assertEqual(set(changed["Property"]), set(DESCRIPTION_REPAIRS))

        structure_cols = ["repo_path", "severity", "tag", "subtag"]
        pd.testing.assert_frame_equal(result[structure_cols], self.base[structure_cols])

        changed_descriptions = result["description"].ne(self.base["description"])
        self.assertEqual(set(result.loc[changed_descriptions, "Property"]), set(DESCRIPTION_REPAIRS))

    def test_repaired_descriptions_are_specific_ascii_single_line(self) -> None:
        result, _ = build_description_quality_repair_v9(self.base)

        for prop in DESCRIPTION_REPAIRS:
            before = self.base.loc[self.base["Property"].eq(prop), "description"].iloc[0]
            after = result.loc[result["Property"].eq(prop), "description"].iloc[0]
            self.assertGreater(len(after), len(str(before)) + 60)
            self.assertNotIn("The finding indicates", after)
            self.assertNotRegex(after, r"[\r\n]")
            after.encode("ascii")

    def test_properties_remain_sequential(self) -> None:
        result, _ = build_description_quality_repair_v9(self.base)

        self.assertEqual(result["Property"].tolist(), list(range(1, 401)))


if __name__ == "__main__":
    unittest.main()
