from __future__ import annotations

import unittest

import pandas as pd

from src.generate_template_description_repair_v4 import (
    DESCRIPTION_REPAIRS,
    build_template_description_repair_v4,
)


class TemplateDescriptionRepairV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.read_csv("outputs/submission_backup_392_template_description_repair_v3.csv")

    def test_repairs_only_configured_descriptions(self) -> None:
        result, changed = build_template_description_repair_v4(self.base)

        self.assertEqual(len(result), 400)
        self.assertFalse(result["repo_path"].astype(str).str.lower().eq("empty").any())
        self.assertEqual(set(changed["Property"]), set(DESCRIPTION_REPAIRS))

        structure_cols = ["repo_path", "severity", "tag", "subtag"]
        pd.testing.assert_frame_equal(result[structure_cols], self.base[structure_cols])

        changed_descriptions = result["description"].ne(self.base["description"])
        self.assertEqual(set(result.loc[changed_descriptions, "Property"]), set(DESCRIPTION_REPAIRS))

    def test_repaired_descriptions_are_specific_ascii_single_line(self) -> None:
        result, _ = build_template_description_repair_v4(self.base)

        for prop in DESCRIPTION_REPAIRS:
            description = result.loc[result["Property"].eq(prop), "description"].iloc[0]
            self.assertGreaterEqual(len(description), 120)
            self.assertNotIn("The finding indicates", description)
            self.assertNotRegex(description, r"[\r\n]")
            description.encode("ascii")

    def test_properties_remain_sequential(self) -> None:
        result, _ = build_template_description_repair_v4(self.base)

        self.assertEqual(result["Property"].tolist(), list(range(1, 401)))


if __name__ == "__main__":
    unittest.main()
