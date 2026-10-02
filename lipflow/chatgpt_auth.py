"""Official Sign in with ChatGPT support for Lipflow GPT RU.

Implements the open-source dynamic client flow documented at
https://developers.openai.com/siwc/token-sharing-open-source/sign-in
using Authorization Code + PKCE and a loopback callback.
Secrets are stored in the OS credential store via keyring; there is no
plaintext fallback.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import keyring
import jwt
import requests
from cryptography.fernet import Fernet, InvalidToken
from jwt import PyJWKClient

from .paths import HOME

AUTH_BASE = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE}/api/accounts/authorize"
TOKEN_URL = f"{AUTH_BASE}/api/accounts/oauth/token"
JWKS_URL = f"{AUTH_BASE}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
ISSUER = AUTH_BASE
DYNAMIC_CLIENT = "dynamic_agent_client"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
AGENT_NAME = "Lipflow GPT RU"
META_PATH = os.path.join(HOME, "chatgpt.json")
SECRETS_PATH = os.path.join(HOME, "chatgpt.secrets")
KEYRING_SERVICE = "Lipflow GPT RU"
KEYRING_USER = "credential-key"


class ChatGPTAuthError(RuntimeError):
    pass


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _atomic_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404)
            return
        self.__class__.result = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        body = (
            "<html><body style='font-family:Segoe UI,sans-serif;padding:40px'>"
            "<h2>Lipflow GPT RU</h2><p>ChatGPT authorization received.</p>"
            "<p>You can close this tab and return to Lipflow.</p></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class ChatGPTSession:
    def __init__(self):
        self.meta = _load_json(META_PATH)
        if not self.meta.get("ext_agent_host_id"):
            self.meta["ext_agent_host_id"] = "urn:uuid:" + str(uuid.uuid4())
            _atomic_json(META_PATH, self.meta)

    def _credential_key(self, create: bool = False) -> bytes | None:
        try:
            raw = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
            if not raw and create:
                raw = Fernet.generate_key().decode("ascii")
                keyring.set_password(KEYRING_SERVICE, KEYRING_USER, raw)
            return raw.encode("ascii") if raw else None
        except Exception as e:
            raise ChatGPTAuthError(f"OS credential store unavailable: {e}") from e

    def _get_secret(self) -> dict:
        if not os.path.exists(SECRETS_PATH):
            return {}
        key = self._credential_key(False)
        if not key:
            raise ChatGPTAuthError("ChatGPT credential key is missing")
        try:
            encrypted = open(SECRETS_PATH, "rb").read()
            return json.loads(Fernet(key).decrypt(encrypted).decode("utf-8"))
        except (OSError, ValueError, InvalidToken) as e:
            raise ChatGPTAuthError("Saved ChatGPT credentials are damaged") from e

    def _save_secret(self, data: dict) -> None:
        key = self._credential_key(True)
        os.makedirs(HOME, exist_ok=True)
        encrypted = Fernet(key).encrypt(json.dumps(data).encode("utf-8"))
        tmp = SECRETS_PATH + ".tmp"
        with open(tmp, "wb") as f:
            f.write(encrypted)
        os.replace(tmp, SECRETS_PATH)

    def connected(self) -> bool:
        try:
            s = self._get_secret()
        except ChatGPTAuthError:
            return False
        return bool(self.meta.get("client_id") and s.get("refresh_token"))

    def label(self) -> str:
        return self.meta.get("email") or ("Connected" if self.connected() else "Not connected")

    def sign_out(self) -> None:
        try:
            if os.path.exists(SECRETS_PATH):
                os.remove(SECRETS_PATH)
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except keyring.errors.PasswordDeleteError:
            pass
        except Exception:
            pass
        host = self.meta.get("ext_agent_host_id")
        self.meta = {"ext_agent_host_id": host}
        _atomic_json(META_PATH, self.meta)

    def sign_in(self, new_account: bool = False, timeout: int = 300) -> dict:
        saved = {} if new_account else self._get_secret()
        saved_client = None if new_account else self.meta.get("client_id")
        client_id = saved_client or DYNAMIC_CLIENT

        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())

        _CallbackHandler.result = None
        server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        server.timeout = timeout
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"

        params = {
            "client_id": client_id,
            "ext_agent_host_id": self.meta["ext_agent_host_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if client_id == DYNAMIC_CLIENT:
            params["agent_name_hint"] = AGENT_NAME
        else:
            if saved.get("id_token"):
                params["id_token_hint"] = saved["id_token"]
            if self.meta.get("email"):
                params["login_hint"] = self.meta["email"]

        if not webbrowser.open(AUTHORIZE_URL + "?" + urlencode(params)):
            raise ChatGPTAuthError("Could not open the system browser for ChatGPT sign-in")
        server.handle_request()
        server.server_close()
        cb = _CallbackHandler.result
        if not cb:
            raise ChatGPTAuthError("ChatGPT sign-in timed out")
        if cb.get("state") != state:
            raise ChatGPTAuthError("ChatGPT sign-in state mismatch")
        if cb.get("error"):
            raise ChatGPTAuthError(f"ChatGPT sign-in was declined: {cb['error']}")
        code = cb.get("code")
        if not code:
            raise ChatGPTAuthError("ChatGPT callback did not contain an authorization code")

        issued_client = cb.get("client_id") or saved_client
        if client_id == DYNAMIC_CLIENT and not issued_client:
            raise ChatGPTAuthError("ChatGPT did not return an issued client ID")
        if saved_client and issued_client and issued_client != saved_client:
            raise ChatGPTAuthError("ChatGPT returned a different client ID for the saved account")

        token = self._exchange_code(issued_client, code, verifier, redirect_uri)
        claims = self._validate_id_token(token.get("id_token", ""), issued_client, nonce)
        granted = set((token.get("scope") or "").split())
        if "chatgpt.tokens.use.direct" not in granted:
            raise ChatGPTAuthError("ChatGPT plan usage permission was not granted")

        token["saved_at"] = time.time()
        self._save_secret(token)
        self.meta.update({
            "client_id": issued_client,
            "subject": claims.get("sub"),
            "email": claims.get("email"),
            "name": claims.get("name"),
            "scopes": sorted(granted),
        })
        _atomic_json(META_PATH, self.meta)
        return dict(self.meta)

    @staticmethod
    def _exchange_code(client_id: str, code: str, verifier: str, redirect_uri: str) -> dict:
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if not r.ok:
            raise ChatGPTAuthError(f"ChatGPT token exchange failed ({r.status_code}): {r.text[:180]}")
        return r.json()

    @staticmethod
    def _validate_id_token(id_token: str, client_id: str, nonce: str) -> dict:
        if not id_token:
            raise ChatGPTAuthError("ChatGPT token response did not include an ID token")
        try:
            signing_key = PyJWKClient(JWKS_URL).get_signing_key_from_jwt(id_token)
            claims = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=client_id,
                issuer=ISSUER,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except Exception as e:
            raise ChatGPTAuthError(f"Could not validate ChatGPT identity token: {e}") from e
        if claims.get("nonce") != nonce:
            raise ChatGPTAuthError("ChatGPT identity token nonce mismatch")
        return claims

    def access_token(self) -> str:
        secret = self._get_secret()
        if not secret or not self.meta.get("client_id"):
            raise ChatGPTAuthError("Connect ChatGPT first")
        saved_at = float(secret.get("saved_at", 0))
        expires_in = int(secret.get("expires_in", 3600))
        if time.time() < saved_at + expires_in - 120:
            return secret["access_token"]
        return self._refresh(secret)

    def _refresh(self, secret: dict) -> str:
        refresh = secret.get("refresh_token")
        if not refresh:
            raise ChatGPTAuthError("ChatGPT session cannot be refreshed; sign in again")
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": self.meta["client_id"],
                "refresh_token": refresh,
                "resource": RESOURCE,
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if not r.ok:
            raise ChatGPTAuthError(f"ChatGPT session refresh failed ({r.status_code}); sign in again")
        fresh = r.json()
        for k in ("id_token", "refresh_token", "access_token", "scope", "expires_in"):
            if k not in fresh and k in secret:
                fresh[k] = secret[k]
        fresh["saved_at"] = time.time()
        self._save_secret(fresh)
        return fresh["access_token"]

    def models(self) -> list[dict]:
        token = self.access_token()
        r = requests.get(
            RESOURCE + "/models",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=20,
        )
        if not r.ok:
            raise ChatGPTAuthError(f"Could not list ChatGPT models ({r.status_code})")
        payload = r.json()
        items = payload.get("models", payload.get("data", []))
        out = []
        for m in items:
            if m.get("visibility") not in (None, "list"):
                continue
            slug = m.get("slug") or m.get("id")
            if slug:
                out.append({"slug": slug, "display_name": m.get("display_name") or slug})
        return out

