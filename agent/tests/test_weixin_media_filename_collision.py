"""Regression: two WeChat file attachments with the same sender-supplied
filename must not overwrite each other on disk.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import src.channels.weixin as weixin_mod
from src.channels.weixin import WeixinChannel

_FAKE_AES_KEY_B64 = "ZmFrZWtleWZha2VrZXkxNg=="


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


class _FakeClient:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(self._payload)


@pytest.fixture()
def channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WeixinChannel:
    monkeypatch.setattr(weixin_mod, "get_media_dir", lambda name: tmp_path)
    monkeypatch.setattr(weixin_mod, "_decrypt_aes_ecb", lambda data, key: data)
    return WeixinChannel({}, bus=None)


def _file_item(encrypt_query_param: str, full_url: str) -> dict:
    return {
        "media": {
            "encrypt_query_param": encrypt_query_param,
            "full_url": full_url,
            "aes_key": _FAKE_AES_KEY_B64,
        },
        "aeskey": "",
    }


class TestFileDownloadFilenameCollision:
    @pytest.mark.parametrize("same_url", [True, False])
    def test_same_second_and_hash_collisions_preserve_both_files(
        self, channel: WeixinChannel, monkeypatch: pytest.MonkeyPatch, same_url: bool
    ) -> None:
        monkeypatch.setattr(weixin_mod.time, "time", lambda: 1000)
        monkeypatch.setattr(weixin_mod, "hash", lambda value: 7, raising=False)

        async def run():
            paths = []
            for index, payload in enumerate((b"first", b"second")):
                channel._client = _FakeClient(payload)
                paths.append(await channel._download_media_item(
                    _file_item("same" if same_url else str(index), "https://cdn/file"),
                    "file", "report.pdf",
                ))
            return paths

        paths = asyncio.run(run())
        assert None not in paths
        assert paths[0] != paths[1]
        assert [Path(path).read_bytes() for path in paths] == [b"first", b"second"]

    def test_same_filename_from_two_messages_does_not_overwrite(
        self, channel: WeixinChannel
    ) -> None:
        async def run() -> tuple[str | None, str | None]:
            channel._client = _FakeClient(b"FIRST FILE CONTENT")
            path1 = await channel._download_media_item(
                _file_item("q1", "https://cdn/1"), "file", "report.pdf"
            )
            channel._client = _FakeClient(b"SECOND FILE CONTENT")
            path2 = await channel._download_media_item(
                _file_item("q2", "https://cdn/2"), "file", "report.pdf"
            )
            return path1, path2

        path1, path2 = asyncio.run(run())

        assert path1 is not None and path2 is not None
        assert path1 != path2
        assert Path(path1).read_bytes() == b"FIRST FILE CONTENT"
        assert Path(path2).read_bytes() == b"SECOND FILE CONTENT"

    def test_unique_name_still_keeps_the_original_extension(
        self, channel: WeixinChannel
    ) -> None:
        async def run() -> str | None:
            channel._client = _FakeClient(b"content")
            return await channel._download_media_item(
                _file_item("q1", "https://cdn/1"), "file", "report.pdf"
            )

        path = asyncio.run(run())

        assert path is not None
        assert path.endswith(".pdf")
        assert Path(path).stem.startswith("report_")
