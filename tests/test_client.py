import ssl
import truststore
from forecaster.reader import MuseClient


def test_muse_client_uses_system_trust_store(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    c = MuseClient()
    ctx = c.ssl_context
    assert isinstance(ctx, truststore.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert str(c._client.base_url).rstrip("/") == "https://api.meta.ai/v1"
