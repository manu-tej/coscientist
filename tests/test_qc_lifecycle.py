from unittest.mock import AsyncMock

import pytest

from qc import hardening, value_experiment


@pytest.mark.parametrize("module", [value_experiment, hardening])
@pytest.mark.asyncio
async def test_qc_entrypoints_close_owned_backend_on_failure(monkeypatch, module):
    backend = object()
    close_backend = AsyncMock()

    monkeypatch.setattr("tools.llm.make_backend", lambda cfg: backend)
    monkeypatch.setattr("tools.llm.close_backend", close_backend)
    monkeypatch.setattr(
        module,
        "run",
        AsyncMock(side_effect=RuntimeError("experiment failed")),
    )

    with pytest.raises(RuntimeError, match="experiment failed"):
        await module._main()

    close_backend.assert_awaited_once_with(backend)
