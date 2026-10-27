"""ZIP upload security: one test per rejection rule, layouts, extraction safety, cleanup."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
import vb_synth
from vb_synth import shapes_zip, write_zip

from visionbench.core.config import Settings
from visionbench.core.errors import DatasetValidationError
from visionbench.data.archive import (
    ArchiveLimits,
    ArchivePlan,
    ArchiveRejected,
    ImportPathError,
    Member,
    extract_archive,
    import_archive,
    import_relpath,
    new_upload_id,
    resolve_import_path,
    upload_path,
    validate_archive,
)
from visionbench.data.scan import ScanConfig

PNG = vb_synth.encode(vb_synth.shape_image(np.random.default_rng(0), "circle"))
LIMITS = ArchiveLimits()


def _rejected(path: Path, code: str, limits: ArchiveLimits = LIMITS) -> ArchiveRejected:
    with pytest.raises(ArchiveRejected) as exc:
        validate_archive(path, limits)
    assert exc.value.code == code, str(exc.value)
    return exc.value


def _zip(tmp_path: Path, names: list[str], **kw) -> Path:
    return write_zip(tmp_path / "a.zip", [("cats/ok.png", PNG), *[(n, PNG) for n in names]], **kw)


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("/etc/passwd.png", "absolute_path"),
        ("C:/Windows/evil.png", "drive_letter"),
        ("c:evil.png", "drive_letter"),
        ("//server/share/evil.png", "unc_path"),
        ("cats/../../evil.png", "parent_traversal"),
        ("../evil.png", "parent_traversal"),
        ("cats/./evil.png", "invalid_name"),
        ("cats//evil.png", "invalid_name"),
        ("cats/ev:il.png", "invalid_name"),
        ("cats/CON.png", "reserved_name"),
        ("cats/nul", "reserved_name"),
        ("cats/com1.txt", "reserved_name"),
        ("LPT9/a.png", "reserved_name"),
        ("cats/a.png.", "trailing_dot_or_space"),
        ("cats/a.png ", "trailing_dot_or_space"),
        ("cats /a.png", "trailing_dot_or_space"),
    ],
)
def test_name_rules(tmp_path: Path, name: str, code: str) -> None:
    err = _rejected(_zip(tmp_path, [name], raw_names=True), code)
    assert err.member == name


@pytest.mark.parametrize("name", ["cats\\..\\..\\evil.png", "..\\evil.png", "cats\\a.png"])
def test_backslash_names_rejected(tmp_path: Path, name: str) -> None:
    _rejected(_zip(tmp_path, [name], raw_names=True), "backslash")


def test_symlink_member(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cats/ok.png", PNG)
        info = zipfile.ZipInfo("cats/link.png")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    _rejected(path, "symlink")


def test_encrypted_member(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "a.zip", [("cats/secret.png", PNG)])
    # zipfile always writes flag_bits = 0, so set the encryption bit in the bytes: general
    # purpose flags sit at offset 6 of the local header and offset 8 of the central entry.
    data = bytearray(path.read_bytes())
    local, central = data.find(b"PK\x03\x04"), data.find(b"PK\x01\x02")
    data[local + 6] |= 0x1
    data[central + 8] |= 0x1
    path.write_bytes(bytes(data))
    assert zipfile.ZipFile(path).infolist()[0].flag_bits & 0x1
    _rejected(path, "encrypted")


def test_member_count_limit(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "a.zip", [(f"cats/{i}.png", PNG) for i in range(6)])
    _rejected(path, "too_many_members", ArchiveLimits(max_members=5))
    validate_archive(path, ArchiveLimits(max_members=6))


def test_total_uncompressed_limit(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "a.zip", [("cats/a.png", b"x" * 3000), ("dogs/b.png", PNG)])
    _rejected(path, "too_large", ArchiveLimits(max_uncompressed_bytes=2000))


def test_member_compression_ratio(tmp_path: Path) -> None:
    path = write_zip(
        tmp_path / "a.zip", [("cats/bomb.png", b"\0" * (2 << 20))], compression=zipfile.ZIP_DEFLATED
    )
    err = _rejected(path, "compression_ratio")
    assert err.member == "cats/bomb.png"


def test_overall_compression_ratio(tmp_path: Path) -> None:
    # each member is below the 1 MiB per-member threshold, but the archive as a whole is a bomb
    members = [(f"cats/{i}.png", b"\0" * (600 << 10)) for i in range(4)]
    path = write_zip(tmp_path / "a.zip", members, compression=zipfile.ZIP_DEFLATED)
    err = _rejected(path, "compression_ratio")
    assert "overall" in str(err) and err.member is None


def test_case_insensitive_collisions(tmp_path: Path) -> None:
    _rejected(
        write_zip(tmp_path / "a.zip", [("Cats/a.png", PNG), ("cats/A.PNG", PNG)]), "case_collision"
    )
    _rejected(write_zip(tmp_path / "b.zip", [("cats", PNG), ("cats/a.png", PNG)]), "case_collision")
    nfc, nfd = "caf\u00e9", "cafe\u0301"
    _rejected(
        write_zip(tmp_path / "c.zip", [(f"{nfc}/a.png", PNG), (f"{nfd}/a.png", PNG)]),
        "case_collision",
    )


def test_magic_bytes_and_corrupt_zip(tmp_path: Path) -> None:
    (tmp_path / "x.zip").write_bytes(b"\x89PNG\r\n\x1a\n" + PNG)
    _rejected(tmp_path / "x.zip", "not_a_zip")
    (tmp_path / "y.zip").write_bytes(b"PK\x03\x04" + b"garbage" * 10)
    _rejected(tmp_path / "y.zip", "corrupt_zip")
    with zipfile.ZipFile(tmp_path / "z.zip", "w"):
        pass
    _rejected(tmp_path / "z.zip", "empty_archive")


def test_layouts(tmp_path: Path) -> None:
    root = validate_archive(shapes_zip(tmp_path / "root.zip"), LIMITS)
    assert root.layout == "class_root" and root.strip_prefix is None
    top = validate_archive(shapes_zip(tmp_path / "top.zip", top="dataset"), LIMITS)
    assert top.layout == "single_top_folder" and top.strip_prefix == "dataset"
    flat = write_zip(tmp_path / "flat.zip", [("a.png", PNG), ("b.png", PNG)])
    assert "class folders" in str(_rejected(flat, "bad_layout"))
    mac = write_zip(
        tmp_path / "mac.zip",
        [("cats/a.png", PNG), ("dogs/b.png", PNG), ("__MACOSX/cats/._a.png", b"x")],
    )
    plan = validate_archive(mac, LIMITS)
    assert plan.ignored == ["__MACOSX/cats/._a.png"] and len(plan.files) == 2


def test_extraction_escape_guard(tmp_path: Path) -> None:
    """Defense in depth: even a plan that bypassed validation cannot write outside staging."""
    path = write_zip(tmp_path / "a.zip", [("cats/a.png", PNG)])
    info = zipfile.ZipFile(path).infolist()[0]
    plan = ArchivePlan([Member(info, ("..", "escaped.png"))], "class_root", None, 10)
    with pytest.raises(ArchiveRejected) as exc:
        extract_archive(path, plan, tmp_path / "staging", LIMITS)
    assert exc.value.code == "extraction_escape"
    assert not (tmp_path / "escaped.png").exists()


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(_env_file=None, data_root=tmp_path / "var", **kw)


def _stage_upload(settings: Settings, source: Path) -> str:
    upload_id = new_upload_id()
    target = upload_path(settings, upload_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return upload_id


def _tmp_listing(settings: Settings) -> list[str]:
    tmp = settings.data_dir("tmp")
    return sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())


def test_import_archive_success_and_cleanup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    nested = shapes_zip(tmp_path / "src.zip", per_class=4, top="mydata")
    with zipfile.ZipFile(nested, "a") as zf:  # a nested archive is stored, never opened
        zf.writestr("mydata/circle/inner.zip", b"PK\x03\x04nested")
    upload_id = _stage_upload(settings, nested)
    r = import_archive(settings, upload_id, "zipped", ScanConfig(), original_filename="src.zip")
    assert r.dataset["class_counts"] == {"circle": 4, "square": 4, "triangle": 4}
    assert r.dataset["source_path"] == "upload:src.zip"
    assert r.report["issues"]["unsupported_extension"]["items"][0]["relpath"] == (
        "circle/inner.zip"
    )
    assert _tmp_listing(settings) == [] and not any(settings.data_dir("tmp").glob("staging-*"))


def test_import_archive_cleans_up_after_rejection_and_failed_import(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bad = _stage_upload(
        settings, write_zip(tmp_path / "bad.zip", [("../x.png", PNG)], raw_names=True)
    )
    with pytest.raises(ArchiveRejected):
        import_archive(settings, bad, "bad", ScanConfig())
    one_class = _stage_upload(
        settings, write_zip(tmp_path / "one.zip", [(f"cats/{i}.png", PNG) for i in range(3)])
    )
    with pytest.raises(DatasetValidationError, match="at least 2 classes"):
        import_archive(settings, one_class, "one", ScanConfig())
    assert _tmp_listing(settings) == []
    assert list(settings.data_dir("tmp").iterdir()) == [settings.data_dir("tmp") / "uploads"]
    assert list(settings.data_dir("datasets").iterdir()) == []


def test_upload_id_validated(tmp_path: Path) -> None:
    with pytest.raises(ArchiveRejected, match="invalid upload id"):
        upload_path(_settings(tmp_path), "../../etc")


def test_path_import_confinement(tmp_path: Path) -> None:
    with pytest.raises(ImportPathError) as exc:
        resolve_import_path(_settings(tmp_path), "anything")
    assert exc.value.code == "path_import_disabled"
    root = tmp_path / "imports"
    vb_synth.make_images(root / "ds" / "a", 3)
    outside = tmp_path / "outside"
    vb_synth.make_images(outside / "a", 3)
    settings = _settings(tmp_path, import_root=root)
    assert resolve_import_path(settings, "ds") == (root / "ds").resolve()
    assert import_relpath(settings, str(root / "ds")) == "ds"
    for bad, code in (
        ("../outside", "path_outside_import_root"),
        (str(outside), "path_outside_import_root"),
        ("missing", "path_not_found"),
    ):
        with pytest.raises(ImportPathError) as exc:
            resolve_import_path(settings, bad)
        assert exc.value.code == code
    link = vb_synth.try_dir_link(outside, root / "link")
    if link is None:
        pytest.skip("cannot create a directory link here")
    with pytest.raises(ImportPathError) as exc:  # junction/symlink resolved before the check
        resolve_import_path(settings, "link")
    assert exc.value.code == "path_outside_import_root"


def test_import_job_handler_via_execute_job(tmp_path: Path) -> None:
    from visionbench.jobs.store import JobStore
    from visionbench.jobs.worker import execute_job
    from visionbench.tracking.store import DB_FILENAME

    settings = _settings(tmp_path)
    good = _stage_upload(settings, shapes_zip(tmp_path / "g.zip", per_class=3))
    bad = _stage_upload(
        settings, write_zip(tmp_path / "b.zip", [("C:/x.png", PNG)], raw_names=True)
    )
    with JobStore(settings.data_dir("db") / DB_FILENAME) as js:
        jobs = [
            js.enqueue(
                "dataset_import", {"source": "upload", "upload_id": u, "name": n}, max_queued=10
            )
            for u, n in ((good, "g"), (bad, "b"))
        ]
        for _ in jobs:
            js.claim(1)
    assert execute_job(jobs[0], settings) == "completed"
    assert execute_job(jobs[1], settings) == "failed"
    with JobStore(settings.data_dir("db") / DB_FILENAME) as js:
        ok, failed = js.get(jobs[0]), js.get(jobs[1])
    assert ok["result"]["total_images"] == 9 and ok["result"]["dataset_id"].startswith("ds-")
    assert failed["error"].startswith("ArchiveRejected: member name starts with a drive letter")
    assert _tmp_listing(settings) == []
