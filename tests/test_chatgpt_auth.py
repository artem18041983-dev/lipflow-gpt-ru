import json

import lipflow.chatgpt_auth as auth


def test_secret_file_is_encrypted(monkeypatch, tmp_path):
    meta = tmp_path / "chatgpt.json"
    secret_file = tmp_path / "chatgpt.secrets"
    store = {}

    monkeypatch.setattr(auth, "META_PATH", str(meta))
    monkeypatch.setattr(auth, "SECRETS_PATH", str(secret_file))
    monkeypatch.setattr(auth.keyring, "get_password", lambda service, user: store.get((service, user)))
    monkeypatch.setattr(auth.keyring, "set_password", lambda service, user, value: store.__setitem__((service, user), value))
    monkeypatch.setattr(auth.keyring, "delete_password", lambda service, user: store.pop((service, user), None))

    session = auth.ChatGPTSession()
    payload = {"access_token": "top-secret-access", "refresh_token": "refresh", "expires_in": 3600}
    session._save_secret(payload)

    encrypted = secret_file.read_bytes()
    assert b"top-secret-access" not in encrypted
    assert session._get_secret() == payload

    saved_meta = json.loads(meta.read_text(encoding="utf-8"))
    assert saved_meta["ext_agent_host_id"].startswith("urn:uuid:")


def test_sign_out_preserves_host_id(monkeypatch, tmp_path):
    meta = tmp_path / "chatgpt.json"
    secret_file = tmp_path / "chatgpt.secrets"
    store = {}

    monkeypatch.setattr(auth, "META_PATH", str(meta))
    monkeypatch.setattr(auth, "SECRETS_PATH", str(secret_file))
    monkeypatch.setattr(auth.keyring, "get_password", lambda service, user: store.get((service, user)))
    monkeypatch.setattr(auth.keyring, "set_password", lambda service, user, value: store.__setitem__((service, user), value))
    monkeypatch.setattr(auth.keyring, "delete_password", lambda service, user: store.pop((service, user), None))

    session = auth.ChatGPTSession()
    host = session.meta["ext_agent_host_id"]
    session._save_secret({"refresh_token": "x"})
    session.meta["client_id"] = "oaiapp_test"
    auth._atomic_json(str(meta), session.meta)

    session.sign_out()
    assert not secret_file.exists()
    assert json.loads(meta.read_text(encoding="utf-8")) == {"ext_agent_host_id": host}

