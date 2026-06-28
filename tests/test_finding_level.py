import unittest

import pandas as pd


class FindingLevelTests(unittest.TestCase):
    def test_scale_counts_hits_target_and_keeps_minimum_one(self):
        from src.finding_utils import scale_counts_to_target

        counts = pd.Series([2, 4, 8], index=["a", "b", "c"])
        scaled = scale_counts_to_target(counts, target_total=30, min_count=1, max_count=20)

        self.assertEqual(int(scaled.sum()), 30)
        self.assertTrue((scaled >= 1).all())
        self.assertTrue((scaled <= 20).all())

    def test_scale_counts_can_hit_exact_400(self):
        from src.finding_utils import scale_counts_to_target

        counts = pd.Series([3, 9, 2, 12, 5] * 11)
        scaled = scale_counts_to_target(counts, target_total=400, min_count=1, max_count=33)

        self.assertEqual(int(scaled.sum()), 400)
        self.assertTrue((scaled >= 1).all())

    def test_variable_length_validator_accepts_multiple_findings_per_repo(self):
        from src.validate_submission import validate_submission_frames

        sample = pd.DataFrame(columns=["Property", "repo_path", "severity", "tag", "subtag", "description"])
        test = pd.DataFrame({"repo_path": ["repo_a", "repo_b"]})
        train = pd.DataFrame(
            {
                "repo_path": ["train_repo"],
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Out of Gas"],
                "description": ["Training description"],
            }
        )
        submission = pd.DataFrame(
            {
                "Property": [1, 2, 3],
                "repo_path": ["repo_a", "repo_a", "repo_b"],
                "severity": ["Medium", "Medium", "Medium"],
                "tag": ["DoS", "DoS", "DoS"],
                "subtag": ["Out of Gas", "Out of Gas", "Out of Gas"],
                "description": [
                    "A specific generated description for repo a.",
                    "Another specific generated description for repo a.",
                    "A specific generated description for repo b.",
                ],
            }
        )

        report = validate_submission_frames(submission, sample, test, train, {"id_column": "Property"}, expected_rows=None)

        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["warnings"], [])

    def test_validator_rejects_non_400_when_expected_rows_set(self):
        from src.validate_submission import validate_submission_frames

        sample = pd.DataFrame(columns=["Property", "repo_path", "severity", "tag", "subtag", "description"])
        test = pd.DataFrame({"repo_path": ["repo_a"]})
        train = pd.DataFrame({"severity": ["Medium"], "tag": ["DoS"], "subtag": ["Out of Gas"]})
        submission = pd.DataFrame(
            {
                "Property": [1],
                "repo_path": ["repo_a"],
                "severity": ["Medium"],
                "tag": ["DoS"],
                "subtag": ["Out of Gas"],
                "description": ["A valid description for a predicted finding."],
            }
        )

        report = validate_submission_frames(submission, sample, test, train, {"id_column": "Property"}, expected_rows=400)

        self.assertFalse(report["passed"])
        self.assertTrue(any("400" in err for err in report["errors"]))

    def test_validate_submission_accepts_explicit_path_argument(self):
        import inspect

        from src import validate_submission

        signature = inspect.signature(validate_submission.main)
        self.assertIn("argv", signature.parameters)

    def test_candidate_generation_outputs_requested_count_and_unique_property(self):
        from src.finding_utils import generate_candidates_for_repos

        train = pd.DataFrame(
            {
                "repo_path": ["r1", "r1", "r2"],
                "severity": ["High", "Medium", "Medium"],
                "tag": ["Access Control", "DoS", "DoS"],
                "subtag": ["Invalid Validation", "Out of Gas", "Out of Gas"],
                "description": ["desc one with enough words", "desc two with enough words", "desc three with enough words"],
            }
        )
        test = pd.DataFrame({"repo_path": ["t1", "t2"]})
        counts = pd.DataFrame({"repo_path": ["t1", "t2"], "pred_count": [2, 1]})

        out = generate_candidates_for_repos(train, test, counts, top_k=2)

        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out["Property"].tolist(), [1, 2, 3])
        self.assertEqual(set(out["repo_path"]), {"t1", "t2"})
        self.assertFalse(out.duplicated(["repo_path", "tag", "subtag", "description"]).any())


if __name__ == "__main__":
    unittest.main()
