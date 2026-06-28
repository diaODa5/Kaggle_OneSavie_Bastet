import io
import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


def make_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


class FetchPublicJudgingTests(unittest.TestCase):
    def test_archive_candidates_are_deterministic_for_github_urls(self):
        from src.fetch_public_judging import archive_candidates

        self.assertEqual(
            archive_candidates("https://github.com/sherlock-audit/2024-01-demo-judging.git"),
            [
                "https://codeload.github.com/sherlock-audit/2024-01-demo-judging/zip/refs/heads/main",
                "https://codeload.github.com/sherlock-audit/2024-01-demo-judging/zip/refs/heads/master",
                "https://github.com/sherlock-audit/2024-01-demo-judging/archive/refs/heads/main.zip",
                "https://github.com/sherlock-audit/2024-01-demo-judging/archive/refs/heads/master.zip",
            ],
        )

    def test_archive_candidates_reject_non_https_and_unapproved_hosts(self):
        from src.fetch_public_judging import archive_candidates

        rejected = [
            "http://github.com/example/demo",
            "file:///tmp/demo.zip",
            "ftp://github.com/example/demo.zip",
            "https://example.test/demo.zip",
            "git@github.com:example/demo.git",
        ]

        self.assertEqual([archive_candidates(url) for url in rejected], [[], [], [], [], []])

    def test_manifest_codeload_candidate_is_fetched_directly(self):
        from src.fetch_public_judging import fetch_contest

        candidate = "https://codeload.github.com/sherlock-audit/demo-judging/zip/refs/heads/main"
        row = {
            "repo_path": "abc123",
            "contest_slug": "demo",
            "candidate_public_url": candidate,
        }
        opened = []

        def opener(url, timeout):
            opened.append(url)
            return io.BytesIO(make_zip({"demo-main/README.md": "# demo"}))

        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_contest(row, Path(tmp), opener=opener)

        self.assertEqual(result["status"], "fetched")
        self.assertEqual(result["url"], candidate)
        self.assertEqual(opened, [candidate])

    def test_task_one_public_judging_url_column_is_supported(self):
        from src.fetch_public_judging import fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "demo",
            "public_judging_url": "https://github.com/sherlock-audit/demo-judging",
        }
        opened = []

        def opener(url, timeout):
            opened.append(url)
            return io.BytesIO(make_zip({"demo-main/README.md": "# demo"}))

        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_contest(row, Path(tmp), opener=opener)

        self.assertEqual(result["status"], "fetched")
        self.assertEqual(
            opened,
            ["https://codeload.github.com/sherlock-audit/demo-judging/zip/refs/heads/main"],
        )

    def test_contest_key_includes_stable_source_hash(self):
        from src.fetch_public_judging import contest_key

        first = {
            "contest_slug": "Same Contest",
            "public_judging_url": "https://github.com/example/first",
        }
        second = {
            "contest_slug": "Same Contest",
            "public_judging_url": "https://github.com/example/second",
        }

        self.assertRegex(contest_key(first), r"^same-contest-[0-9a-f]{12}$")
        self.assertEqual(contest_key(first), contest_key(dict(first)))
        self.assertNotEqual(contest_key(first), contest_key(second))

    def test_run_fetch_rejects_duplicate_source_in_same_manifest(self):
        from src.fetch_public_judging import run_fetch

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        manifest = (
            "contest_slug,public_judging_url\n"
            f"demo,{url}\n"
            f"demo-copy,{url}\n"
        )
        opened = []

        def opener(candidate, timeout):
            opened.append(candidate)
            return io.BytesIO(make_zip({"repo/README.md": "# demo"}))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(manifest, encoding="utf-8")

            results = run_fetch(
                manifest_path,
                root / "cache",
                root / "report.md",
                opener=opener,
            )

        self.assertEqual([result["status"] for result in results], ["fetched", "failed"])
        self.assertIn("duplicate source", results[1]["error"])
        self.assertEqual(opened, [url])

    def test_oversized_download_is_rejected_before_it_can_be_cached(self):
        from src.fetch_public_judging import MAX_ARCHIVE_BYTES, contest_key, fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "demo",
            "candidate_public_url": "https://codeload.github.com/example/demo/zip/main",
        }

        class OversizedResponse:
            def read(self, size):
                return b"x" * size

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fetch_contest(row, root, opener=lambda url, timeout: OversizedResponse())

            self.assertEqual(result["status"], "failed")
            self.assertIn("size limit", result["attempts"][0]["error"])
            self.assertFalse((root / "archives" / f"{contest_key(row)}.zip").exists())
            self.assertEqual(MAX_ARCHIVE_BYTES, 50 * 1024 * 1024)

    def test_download_loops_until_eof_when_response_short_reads(self):
        from src.fetch_public_judging import fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        class ShortReadResponse:
            headers = {}

            def __init__(self):
                self.buffer = io.BytesIO(payload)

            def read(self, size):
                return self.buffer.read(min(size, 7))

            def close(self):
                self.buffer.close()

        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_contest(
                row,
                Path(tmp),
                opener=lambda candidate, timeout: ShortReadResponse(),
            )

        self.assertEqual(result["status"], "fetched")

    def test_download_rejects_content_length_mismatch(self):
        from src.fetch_public_judging import fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        class TruncatedResponse(io.BytesIO):
            headers = {"Content-Length": str(len(payload) + 10)}

        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_contest(
                row,
                Path(tmp),
                opener=lambda candidate, timeout: TruncatedResponse(payload),
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("Content-Length", result["attempts"][0]["error"])

    def test_download_deadline_covers_all_response_reads(self):
        from src.fetch_public_judging import fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}

        class ChunkedResponse:
            headers = {}

            def read(self, size):
                return b"x"

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("src.fetch_public_judging.time", create=True) as fake_time:
                fake_time.monotonic.side_effect = [100.0, 100.5, 101.1]
                result = fetch_contest(
                    row,
                    Path(tmp),
                    opener=lambda candidate, timeout: ChunkedResponse(),
                    timeout=1,
                )

        self.assertEqual(result["status"], "failed")
        self.assertIn("deadline", result["attempts"][0]["error"])

    def test_fetch_rejects_non_positive_timeout_before_opening_network(self):
        from src.fetch_public_judging import fetch_contest

        row = {
            "contest_slug": "demo",
            "public_judging_url": "https://codeload.github.com/example/demo/zip/main",
        }

        for timeout in (0, -1):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "timeout must be positive"):
                    fetch_contest(
                        row,
                        Path(tmp),
                        opener=lambda *args: self.fail("opener must not be called"),
                        timeout=timeout,
                    )

    def test_fetch_uses_cached_archive_without_opening_network(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "2024-01-demo",
            "judging_url": "https://github.com/sherlock-audit/2024-01-demo-judging",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"
            archive.parent.mkdir(parents=True)
            payload = make_zip({"repo-main/README.md": "# cached"})
            archive.write_bytes(payload)
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "url": row["judging_url"],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            def forbidden_opener(*args, **kwargs):
                raise AssertionError("network opener must not be called for a cached archive")

            result = fetch_contest(row, root, opener=forbidden_opener)

            self.assertEqual(result["status"], "cached")
            self.assertEqual(result["source"], "archive_cache")
            self.assertTrue((root / "extracted" / contest_key(row) / "repo-main" / "README.md").exists())

    def test_network_fetch_writes_source_and_sha256_metadata(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fetch_contest(row, root, opener=lambda candidate, timeout: io.BytesIO(payload))
            metadata_path = root / "archives" / f"{contest_key(row)}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "fetched")
        self.assertEqual(metadata["url"], url)
        self.assertEqual(metadata["sha256"], hashlib.sha256(payload).hexdigest())

    def test_cached_archive_rejects_sha256_mismatch(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        original = make_zip({"repo/README.md": "# original"})
        tampered = make_zip({"repo/README.md": "# tampered"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(tampered)
            archive.with_suffix(".json").write_text(
                json.dumps({"url": url, "sha256": hashlib.sha256(original).hexdigest()}),
                encoding="utf-8",
            )

            result = fetch_contest(
                row,
                root,
                opener=lambda *args: self.fail("invalid cache must not fall through to network"),
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("SHA256", result["error"])

    def test_cached_archive_rejects_missing_metadata(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        row = {
            "contest_slug": "demo",
            "public_judging_url": "https://github.com/example/demo",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(make_zip({"repo/README.md": "# demo"}))

            result = fetch_contest(row, root, opener=lambda *args: self.fail("no network"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("metadata", result["error"])

    def test_cached_archive_rejects_unreadable_zip_with_matching_sha256(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = b"not a ZIP archive"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(payload)
            archive.with_suffix(".json").write_text(
                json.dumps({"url": url, "sha256": hashlib.sha256(payload).hexdigest()}),
                encoding="utf-8",
            )

            result = fetch_contest(row, root, opener=lambda *args: self.fail("no network"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("BadZipFile", result["error"])

    def test_cached_archive_rejects_empty_existing_extraction(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = contest_key(row)
            archive = root / "archives" / f"{key}.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(payload)
            archive.with_suffix(".json").write_text(
                json.dumps({"url": url, "sha256": hashlib.sha256(payload).hexdigest()}),
                encoding="utf-8",
            )
            (root / "extracted" / key).mkdir(parents=True)

            result = fetch_contest(row, root, opener=lambda *args: self.fail("no network"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("extraction is empty", result["error"])

    def test_cached_archive_rejects_source_url_mismatch(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(payload)
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "url": "https://codeload.github.com/example/other/zip/refs/heads/main",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            result = fetch_contest(row, root, opener=lambda *args: self.fail("no network"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("source URL", result["error"])

    def test_valid_zip_without_metadata_files_is_rejected(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fetch_contest(
                row,
                root,
                opener=lambda candidate, timeout: io.BytesIO(
                    make_zip({"repo/src/Contract.sol": "contract Contract {}"})
                ),
            )

            self.assertEqual(result["status"], "failed")
            self.assertIn("no extractable metadata", result["attempts"][0]["error"])
            self.assertFalse((root / "archives" / f"{contest_key(row)}.zip").exists())
            self.assertFalse((root / "extracted" / contest_key(row)).exists())

    def test_fetch_uses_local_dropin_before_network_and_preserves_dropin(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "2024-01-demo",
            "judging_url": "https://github.com/sherlock-audit/2024-01-demo-judging",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dropin = root / "dropins" / f"{contest_key(row)}.zip"
            dropin.parent.mkdir(parents=True)
            payload = make_zip({"repo-main/issues/1.md": "# issue"})
            dropin.write_bytes(payload)

            def forbidden_opener(*args, **kwargs):
                raise AssertionError("network opener must not be called when a drop-in exists")

            result = fetch_contest(row, root, opener=forbidden_opener)

            self.assertEqual(result["status"], "fetched")
            self.assertEqual(result["source"], "local_dropin")
            self.assertEqual(dropin.read_bytes(), payload)
            self.assertEqual(
                (root / "archives" / f"{contest_key(row)}.zip").read_bytes(),
                payload,
            )

    def test_fetch_records_every_failed_download_attempt(self):
        from src.fetch_public_judging import archive_candidates, fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "2024-01-demo",
            "judging_url": "https://github.com/sherlock-audit/2024-01-demo-judging",
        }
        attempted = []

        def failing_opener(url, timeout):
            attempted.append((url, timeout))
            raise OSError("network unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_contest(row, Path(tmp), opener=failing_opener, timeout=3)

        expected_urls = archive_candidates(row["judging_url"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual([item["url"] for item in result["attempts"]], expected_urls)
        self.assertTrue(all("network unavailable" in item["error"] for item in result["attempts"]))
        self.assertEqual(attempted, [(url, 3) for url in expected_urls])

    def test_safe_extraction_rejects_path_traversal_without_writing_files(self):
        from src.fetch_public_judging import UnsafeArchiveError, extract_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            archive.write_bytes(
                make_zip(
                    {
                        "repo/issue.md": "safe",
                        "../escaped.md": "unsafe",
                    }
                )
            )
            destination = root / "out"

            with self.assertRaises(UnsafeArchiveError):
                extract_metadata(archive, destination)

            self.assertFalse(destination.exists())
            self.assertFalse((root / "escaped.md").exists())

    def test_safe_extraction_rejects_symbolic_link_member(self):
        from src.fetch_public_judging import UnsafeArchiveError, extract_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "symlink.zip"
            link = zipfile.ZipInfo("repo/link.md")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(link, "../outside.md")

            destination = root / "out"
            with self.assertRaisesRegex(UnsafeArchiveError, "symbolic link"):
                extract_metadata(archive, destination)

            self.assertFalse(destination.exists())

    def test_safe_extraction_selects_only_markdown_json_and_text_metadata(self):
        from src.fetch_public_judging import extract_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fixture.zip"
            archive.write_bytes(
                make_zip(
                    {
                        "repo/issues/1.md": "# finding",
                        "repo/data/1.JSON": '{"severity": "High"}',
                        "repo/notes.txt": "note",
                        "repo/src/Contract.sol": "contract Contract {}",
                        "repo/image.png": "not metadata",
                    }
                )
            )

            summary = extract_metadata(archive, root / "out")

            extracted = sorted(path.relative_to(root / "out").as_posix() for path in (root / "out").rglob("*") if path.is_file())
            self.assertEqual(extracted, ["repo/data/1.JSON", "repo/issues/1.md", "repo/notes.txt"])
            self.assertEqual(summary["extracted_files"], 3)
            self.assertEqual(summary["skipped_files"], 2)

    def test_invalid_download_is_not_cached_and_next_candidate_can_succeed(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        row = {
            "repo_path": "abc123",
            "contest_slug": "2024-01-demo",
            "judging_url": "https://github.com/sherlock-audit/2024-01-demo-judging",
        }
        valid = make_zip({"repo-main/README.md": "# ok"})
        responses = iter([b"not a zip", valid])

        def opener(url, timeout):
            return io.BytesIO(next(responses))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = fetch_contest(row, root, opener=opener)
            archive = root / "archives" / f"{contest_key(row)}.zip"

            self.assertEqual(result["status"], "fetched")
            self.assertEqual(len(result["attempts"]), 2)
            self.assertIn("zip", result["attempts"][0]["error"].lower())
            self.assertEqual(archive.read_bytes(), valid)

    def test_archive_publish_failure_removes_committed_extraction(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# demo"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("os.link", side_effect=OSError("commit failed")):
                result = fetch_contest(
                    row,
                    root,
                    opener=lambda candidate, timeout: io.BytesIO(payload),
                )

            key = contest_key(row)
            self.assertEqual(result["status"], "failed")
            self.assertIn("commit failed", result["attempts"][0]["error"])
            self.assertFalse((root / "archives" / f"{key}.zip").exists())
            self.assertFalse((root / "archives" / f"{key}.json").exists())
            self.assertFalse((root / "extracted" / key).exists())

    def test_archive_publish_does_not_overwrite_racing_destination(self):
        from src.fetch_public_judging import contest_key, fetch_contest

        url = "https://codeload.github.com/example/demo/zip/refs/heads/main"
        row = {"contest_slug": "demo", "public_judging_url": url}
        payload = make_zip({"repo/README.md": "# downloaded"})
        sentinel = b"existing archive from another process"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / f"{contest_key(row)}.zip"

            def racing_opener(candidate, timeout):
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(sentinel)
                return io.BytesIO(payload)

            result = fetch_contest(row, root, opener=racing_opener)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(archive.read_bytes(), sentinel)
            self.assertFalse((root / "extracted" / contest_key(row)).exists())

    def test_existing_extraction_is_never_overwritten(self):
        from src.fetch_public_judging import extract_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fixture.zip"
            archive.write_bytes(make_zip({"repo/issue.md": "new"}))
            destination = root / "out"
            existing = destination / "repo" / "issue.md"
            existing.parent.mkdir(parents=True)
            existing.write_text("old", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                extract_metadata(archive, destination)

            self.assertEqual(existing.read_text(encoding="utf-8"), "old")

    def test_missing_manifest_writes_clear_unavailable_report(self):
        from src.fetch_public_judging import run_fetch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.md"

            results = run_fetch(root / "missing.csv", root / "cache", report)

            self.assertEqual(results, [])
            text = report.read_text(encoding="utf-8")
            self.assertIn("status: `unavailable`", text)
            self.assertIn("manifest not found", text)

    def test_report_summarizes_fetched_cached_unavailable_and_failed(self):
        from src.fetch_public_judging import write_report

        results = [
            {"key": "a", "contest_slug": "a", "status": "fetched", "source": "network", "url": "https://a", "attempts": []},
            {"key": "b", "contest_slug": "b", "status": "cached", "source": "archive_cache", "url": "", "attempts": []},
            {"key": "c", "contest_slug": "c", "status": "unavailable", "source": "", "url": "", "error": "no judging URL", "attempts": []},
            {
                "key": "d",
                "contest_slug": "d",
                "status": "failed",
                "source": "",
                "url": "",
                "attempts": [{"url": "https://d", "error": "offline"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            write_report(results, report, manifest_path=Path("manifest.csv"))
            text = report.read_text(encoding="utf-8")

        self.assertIn("- fetched: `1`", text)
        self.assertIn("- cached: `1`", text)
        self.assertIn("- unavailable: `1`", text)
        self.assertIn("- failed: `1`", text)
        self.assertIn("https://d", text)
        self.assertIn("offline", text)


if __name__ == "__main__":
    unittest.main()
