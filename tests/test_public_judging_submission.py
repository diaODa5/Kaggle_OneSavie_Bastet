import unittest

import pandas as pd


class PublicJudgingSubmissionTests(unittest.TestCase):
    def test_precision_policy_pads_to_400_and_keeps_sequence(self):
        from src.generate_public_judging_submission import build_candidate_submission

        sample = pd.DataFrame(columns=["Property", "repo_path", "severity", "tag", "subtag", "description"])
        test = pd.DataFrame({"repo_path": ["repo_a"]})
        train = pd.DataFrame(
            {
                "severity": ["Medium"],
                "tag": ["Arithmetic"],
                "subtag": ["Precision Loss"],
            }
        )
        findings = pd.DataFrame(
            {
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["Arithmetic"],
                "subtag": ["Precision Loss"],
                "description": ["Rounding in the fee calculation can make pairs revert for maximum fee settings."],
                "confidence": ["exact"],
                "mapping_method": ["external_exact"],
                "mapping_ambiguity": [False],
                "labels_complete": [True],
            }
        )

        out, provenance = build_candidate_submission(findings, sample, test, train, "precision")

        self.assertEqual(out.shape, (400, 6))
        self.assertEqual(out["Property"].tolist(), list(range(1, 401)))
        self.assertEqual(out.iloc[0]["repo_path"], "repo_a")
        self.assertEqual(out.iloc[1]["repo_path"], "empty")
        self.assertEqual(len(provenance), 400)

    def test_balanced_policy_adds_manual_sherlock_labels(self):
        from src.generate_public_judging_submission import apply_manual_sherlock_labels

        train = pd.DataFrame(
            {
                "severity": ["Medium"],
                "tag": ["Arithmetic"],
                "subtag": ["Precision Loss"],
            }
        )
        findings = pd.DataFrame(
            {
                "repo_path": ["103f39b0f29b"],
                "contest": ["2024-02-rubicon-finance"],
                "issue_id": ["001-M"],
                "title": ['Pairs with "MAX_FEE" can revert due to rounding inconsistencies'],
                "severity": ["Medium"],
                "tag": [""],
                "subtag": [""],
                "description": ["Pairs with MAX_FEE can revert because fee rounding is inconsistent."],
                "confidence": ["exact"],
                "mapping_method": ["unresolved"],
                "mapping_ambiguity": [False],
                "labels_complete": [False],
            }
        )

        out = apply_manual_sherlock_labels(findings, train)

        self.assertTrue(out.iloc[0]["labels_complete"])
        self.assertEqual(out.iloc[0]["tag"], "Arithmetic")
        self.assertEqual(out.iloc[0]["subtag"], "Precision Loss")
        self.assertEqual(out.iloc[0]["mapping_method"], "manual_sherlock")

    def test_candidate_builder_rejects_illegal_labels(self):
        from src.generate_public_judging_submission import build_candidate_submission

        sample = pd.DataFrame(columns=["Property", "repo_path", "severity", "tag", "subtag", "description"])
        test = pd.DataFrame({"repo_path": ["repo_a"]})
        train = pd.DataFrame({"severity": ["Medium"], "tag": ["DoS"], "subtag": ["Out of Gas"]})
        findings = pd.DataFrame(
            {
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["Not Legal"],
                "subtag": ["Out of Gas"],
                "description": ["A specific issue with enough descriptive context."],
                "confidence": ["exact"],
                "mapping_method": ["external_exact"],
                "mapping_ambiguity": [False],
                "labels_complete": [True],
            }
        )

        with self.assertRaises(ValueError):
            build_candidate_submission(findings, sample, test, train, "balanced")


if __name__ == "__main__":
    unittest.main()
