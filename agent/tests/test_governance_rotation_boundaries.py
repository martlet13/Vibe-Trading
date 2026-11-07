"""Rotation preserves sealed bytes and refuses corrupt historical segments."""

import os

import pytest

from src.governance.ledger import (
    LedgerCorruptionError,
    append_record,
    archive_segments,
    rotate_if_needed,
    verify_chain_with_archives,
)


@pytest.mark.parametrize("number", [2, 9999])
def test_number_gaps_and_counter_width_do_not_overwrite_history(tmp_path, number):
    path = tmp_path / "audit.jsonl"
    append_record(path, {"value": 1})
    first = rotate_if_needed(path, 1)
    renamed = path.with_name(f"audit.{number:04d}.jsonl")
    first.rename(renamed)
    sealed = renamed.read_bytes()
    append_record(path, {"value": 2})
    second = rotate_if_needed(path, 1)
    assert second.name == f"audit.{number + 1:04d}.jsonl"
    assert renamed.read_bytes() == sealed
    append_record(path, {"value": 3})
    assert archive_segments(path) == [renamed, second]
    result = verify_chain_with_archives(path)
    assert result.ok and result.record_count == 3


def test_rotation_refuses_corruption_in_sealed_history(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_record(path, {"value": 1})
    sealed = rotate_if_needed(path, 1)
    append_record(path, {"value": 2})
    sealed.write_text(sealed.read_text().replace('"value": 1', '"value": 9'))
    active = path.read_bytes()
    with pytest.raises(LedgerCorruptionError):
        rotate_if_needed(path, 1)
    assert path.read_bytes() == active
    assert archive_segments(path) == [sealed]


def test_racing_archive_destination_is_never_replaced(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    append_record(path, {"value": 1})
    before = path.read_bytes()
    link = os.link

    def collision(source, destination):
        destination.write_bytes(b"existing sealed bytes")
        link(source, destination)

    monkeypatch.setattr(os, "link", collision)
    with pytest.raises(FileExistsError):
        rotate_if_needed(path, 1)
    assert path.read_bytes() == before
    assert (tmp_path / "audit.0001.jsonl").read_bytes() == b"existing sealed bytes"
