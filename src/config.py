from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_KAGGLE_DIR = DATA_DIR / "raw_kaggle"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
FILE_LIST_PATH = DATA_DIR / "file_list.txt"

WORKING_DIR = PROJECT_ROOT / "working"
SCHEMA_SUMMARY_PATH = WORKING_DIR / "schema_summary.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDS_DIR = OUTPUTS_DIR / "preds"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DATA_REPORT_PATH = REPORTS_DIR / "data_report.md"
PREPARE_REPORT_PATH = REPORTS_DIR / "prepare_report.md"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_report.md"
MODEL_REPORT_PATH = REPORTS_DIR / "model_report.md"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"
SUBMISSION_DIAGNOSIS_REPORT_PATH = REPORTS_DIR / "submission_diagnosis.md"
FINDING_LEVEL_TASK_DIAGNOSIS_PATH = REPORTS_DIR / "finding_level_task_diagnosis.md"
FINDING_COUNT_REPORT_PATH = REPORTS_DIR / "finding_count_report.md"
FINDING_GENERATION_OOF_REPORT_PATH = REPORTS_DIR / "finding_generation_oof_report.md"

FINDING_COUNTS_CONSERVATIVE_PATH = PREDS_DIR / "finding_counts_conservative.csv"
FINDING_COUNTS_BALANCED_PATH = PREDS_DIR / "finding_counts_balanced.csv"
FINDING_COUNTS_AGGRESSIVE_PATH = PREDS_DIR / "finding_counts_aggressive.csv"
FINDING_COUNTS_400_PATH = PREDS_DIR / "finding_counts_400.csv"
CANDIDATE_FINDINGS_BALANCED_PATH = PREDS_DIR / "candidate_findings_balanced.csv"
CANDIDATE_FINDINGS_400_PATH = PREDS_DIR / "candidate_findings_400.csv"
FINDING_GENERATION_OOF_PATH = PREDS_DIR / "finding_generation_oof.csv"

SUBMISSION_ONE_PER_REPO_PATH = OUTPUTS_DIR / "submission_one_per_repo.csv"
SUBMISSION_CONSERVATIVE_PATH = OUTPUTS_DIR / "submission_conservative.csv"
SUBMISSION_BALANCED_PATH = OUTPUTS_DIR / "submission_balanced.csv"
SUBMISSION_AGGRESSIVE_PATH = OUTPUTS_DIR / "submission_aggressive.csv"
SELECTED_SUBMISSION_INFO_PATH = OUTPUTS_DIR / "selected_submission_info.json"
EVALUATION_RULES_REPORT_PATH = REPORTS_DIR / "evaluation_and_rules_report.md"
EXTERNAL_BASTET_REPORT_PATH = REPORTS_DIR / "external_bastet_report.md"
TEST_EXTERNAL_MATCH_REPORT_PATH = REPORTS_DIR / "test_external_match_report.md"
LABEL_MAPPING_REPORT_PATH = REPORTS_DIR / "label_mapping_report.md"
EXTERNAL_SUBMISSION_REPORT_PATH = REPORTS_DIR / "external_submission_report.md"
EXTERNAL_OOF_REPORT_PATH = REPORTS_DIR / "external_oof_report.md"
EXTERNAL_ZIP_REPORT_PATH = REPORTS_DIR / "external_zip_report.md"
EXTERNAL_EXTRACT_REPORT_PATH = REPORTS_DIR / "external_extract_report.md"
DESCRIPTION_ENHANCEMENT_REPORT_PATH = REPORTS_DIR / "description_enhancement_report.md"
REPO_HASH_DIAGNOSIS_REPORT_PATH = REPORTS_DIR / "repo_hash_diagnosis_report.md"
TRAIN_EXTERNAL_FINDING_MATCH_REPORT_PATH = REPORTS_DIR / "train_external_finding_match_report.md"
HASH_ALGORITHM_REPORT_PATH = REPORTS_DIR / "hash_algorithm_report.md"
TEST_HASH_FILE_SEARCH_REPORT_PATH = REPORTS_DIR / "test_hash_external_file_search_report.md"
FINAL_TEST_REPO_MAPPING_REPORT_PATH = REPORTS_DIR / "final_test_repo_mapping_report.md"
EVOLVED_SUBMISSION_REPORT_PATH = REPORTS_DIR / "evolved_submission_report.md"
EXTERNAL_FINGERPRINT_REPORT_PATH = REPORTS_DIR / "external_fingerprint_report.md"
EXTERNAL_REPO_FINGERPRINTS_PATH = PROCESSED_DIR / "external_repo_fingerprints.parquet"
FINGERPRINT_HASH_RULE_CANDIDATES_PATH = PROCESSED_DIR / "fingerprint_hash_rule_candidates.csv"
TEST_HASH_TO_EXTERNAL_BY_FINGERPRINT_PATH = PROCESSED_DIR / "test_hash_to_external_repo_by_fingerprint.csv"
FINGERPRINT_HASH_INFERENCE_REPORT_PATH = REPORTS_DIR / "fingerprint_hash_inference_report.md"

EXTERNAL_BASTET_FINDINGS_PATH = PROCESSED_DIR / "external_bastet_findings.parquet"
EXTERNAL_BASTET_MAPPED_FINDINGS_PATH = PROCESSED_DIR / "external_bastet_findings_mapped.parquet"
TEST_EXTERNAL_MATCHES_PATH = PROCESSED_DIR / "test_external_matches.parquet"
EXTERNAL_LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping_external_to_kaggle.json"
SUBMISSION_EXTERNAL_MATCHED_PATH = OUTPUTS_DIR / "submission_external_matched.csv"
EXTERNAL_CANDIDATE_FINDINGS_400_PATH = PREDS_DIR / "external_candidate_findings_400.csv"
EXTERNAL_CANDIDATE_FINDINGS_400_ENHANCED_PATH = PREDS_DIR / "external_candidate_findings_400_enhanced.csv"
TRAIN_HASH_EXTERNAL_CANDIDATES_PATH = PROCESSED_DIR / "train_hash_to_external_repo_candidates.parquet"
HIGH_CONF_TRAIN_HASH_EXTERNAL_PAIRS_PATH = PROCESSED_DIR / "high_confidence_train_hash_external_pairs.csv"
INFERRED_HASH_RULE_PATH = PROCESSED_DIR / "inferred_hash_rule.json"
TEST_HASH_TO_EXTERNAL_BY_HASH_PATH = PROCESSED_DIR / "test_hash_to_external_repo_by_hash_algorithm.csv"
TEST_HASH_FILE_OCCURRENCES_PATH = PROCESSED_DIR / "test_hash_file_occurrences.parquet"
FINAL_TEST_HASH_MAPPING_PATH = PROCESSED_DIR / "final_test_hash_to_external_repo_mapping.csv"
SUBMISSION_FALLBACK_ONLY_PATH = OUTPUTS_DIR / "submission_fallback_only.csv"
SUBMISSION_MAPPED_EXTERNAL_PATH = OUTPUTS_DIR / "submission_mapped_external.csv"
SUBMISSION_HIGH_PRECISION_PATH = OUTPUTS_DIR / "submission_high_precision.csv"
FINGERPRINT_SUBMISSION_REPORT_PATH = REPORTS_DIR / "fingerprint_submission_report.md"
SUBMISSION_FINGERPRINT_MATCHED_PATH = OUTPUTS_DIR / "submission_fingerprint_matched.csv"
SUBMISSION_FINGERPRINT_HIGH_PRECISION_PATH = OUTPUTS_DIR / "submission_fingerprint_high_precision.csv"
SUBMISSION_FINGERPRINT_RECALL_PATH = OUTPUTS_DIR / "submission_fingerprint_recall.csv"
SUBMISSION_TRAIN_PRIOR_PATH = OUTPUTS_DIR / "submission_train_prior_400.csv"
SUBMISSION_COUNT_SPREAD_PATH = OUTPUTS_DIR / "submission_count_spread_400.csv"
SUBMISSION_HIGHFREQ_TUPLE_PATH = OUTPUTS_DIR / "submission_highfreq_tuple_400.csv"
SUBMISSION_BASELINE_CLEAN_DESC_PATH = OUTPUTS_DIR / "submission_baseline_clean_desc_400.csv"
SUBMISSION_EVOLVED_RECOMMENDED_PATH = OUTPUTS_DIR / "submission_evolved_recommended.csv"

EXTERNAL_ZIP_PATH = EXTERNAL_DIR / "dataset_v0.zip"
EXTERNAL_BASTET_V0_DIR = EXTERNAL_DIR / "bastet_v0"

TRAIN_CSV_PATH = RAW_KAGGLE_DIR / "train.csv"
TEST_CSV_PATH = RAW_KAGGLE_DIR / "test.csv"
SUBMISSION_EXAMPLE_PATH = RAW_KAGGLE_DIR / "submission_example.csv"
OFFICIAL_TRAIN_ZIP_PATH = RAW_KAGGLE_DIR / "train.zip"
OFFICIAL_TEST_ZIP_PATH = RAW_KAGGLE_DIR / "test.zip"
OFFICIAL_REPO_FINGERPRINTS_PATH = PROCESSED_DIR / "official_repo_fingerprints.parquet"
OFFICIAL_EXTERNAL_REPO_MATCHES_PATH = PROCESSED_DIR / "official_external_repo_matches.parquet"
OFFICIAL_REPO_FINGERPRINT_REPORT_PATH = REPORTS_DIR / "official_repo_fingerprint_report.md"
OFFICIAL_EXTERNAL_MATCH_REPORT_PATH = REPORTS_DIR / "official_external_match_report.md"
UNMATCHED_TEST_REPO_IDENTITY_PATH = PROCESSED_DIR / "unmatched_test_repo_identity.parquet"
UNMATCHED_TEST_REPO_IDENTITY_REPORT_PATH = REPORTS_DIR / "unmatched_test_repo_identity_report.md"
UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH = PROCESSED_DIR / "unmatched_test_repo_soft_matches.parquet"
UNMATCHED_TEST_REPO_SOFT_MATCH_REPORT_PATH = REPORTS_DIR / "unmatched_test_repo_soft_match_report.md"
UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH = PROCESSED_DIR / "unmatched_test_repo_file_overlap_matches.parquet"
UNMATCHED_TEST_REPO_FILE_OVERLAP_REPORT_PATH = REPORTS_DIR / "unmatched_test_repo_file_overlap_match_report.md"
SUBMISSION_OFFICIAL_MATCHED_PATH = OUTPUTS_DIR / "submission_official_matched.csv"
SUBMISSION_OFFICIAL_HIGH_PRECISION_PATH = OUTPUTS_DIR / "submission_official_high_precision.csv"
SUBMISSION_OFFICIAL_EXPANDED_COVERAGE_PATH = OUTPUTS_DIR / "submission_official_expanded_coverage.csv"
SUBMISSION_OFFICIAL_EXPANDED_PLUS_OPTIMISM_PATH = OUTPUTS_DIR / "submission_official_expanded_plus_optimism.csv"
OFFICIAL_SUBMISSION_REPORT_PATH = REPORTS_DIR / "official_submission_report.md"

TRAIN_PROCESSED_PATH = PROCESSED_DIR / "train_processed.parquet"
TEST_PROCESSED_PATH = PROCESSED_DIR / "test_processed.parquet"
FEATURE_INFO_PATH = PROCESSED_DIR / "feature_info.json"
TARGET_INFO_PATH = PROCESSED_DIR / "target_info.json"

BASELINE_OOF_PATH = PREDS_DIR / "baseline_oof.parquet"
BASELINE_TEST_PATH = PREDS_DIR / "baseline_test.parquet"
MODEL_OOF_PATH = PREDS_DIR / "model_oof.parquet"
MODEL_TEST_PATH = PREDS_DIR / "model_test.parquet"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
FINAL_MODELS_PATH = MODELS_DIR / "final_models.pkl"

OUTPUT_SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"
ROOT_SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_KAGGLE_DIR,
        EXTERNAL_DIR,
        PROCESSED_DIR,
        WORKING_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        PREDS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
