# Lipflow GPT RU

Experimental Windows fork of Lipflow with official **Sign in with ChatGPT** and Russian/English visual dictation.

## What changed

- Official OpenAI open-source OAuth flow (`dynamic_agent_client`, Authorization Code + PKCE).
- Uses eligible ChatGPT Plus/Pro plan allowance; no API key is required.
- OAuth tokens are encrypted locally. The encryption key is kept in the OS credential store.
- Responses API calls use `store: false` and `stream: true`.
- Silent lip reading supports Russian, English, and Auto RU/EN modes.
- Only aligned mouth crops are sent to the model, packed into chronological contact sheets.
- Dictation mouth clips are not retained locally by default.
- Original Lipflow English VSR remains available as a fallback engine during the prototype phase.

## Privacy boundary

The ChatGPT mode sends:
1. sampled aligned mouth crops for the current utterance;
2. up to the last three dictated phrases as ambiguity context;
3. names/terms inferred from the active window title.

It does **not** receive ChatGPT conversations, ChatGPT memory, API keys, or arbitrary screen contents.

## Windows UX

Tray menu:
- Continue with ChatGPT
- Disconnect ChatGPT
- Language → Russian / English / Auto RU / EN
- Recognition engine → ChatGPT Vision / Original Lipflow (English)
- Push-to-talk key
- Camera

Default:
- engine: ChatGPT Vision
- language: Russian
- saved mouth clips: off

## Status

Prototype. Unit-tested on Linux ARM for the new OAuth/vision modules.
Full application testing is performed on Windows x64 because the upstream MediaPipe build has no Linux ARM64 wheel.

## OpenAI contract used

- Authorization: https://auth.openai.com/api/accounts/authorize
- Token exchange/refresh: https://auth.openai.com/api/accounts/oauth/token
- Model discovery: https://api.openai.com/v1/models
- Inference: https://api.openai.com/v1/responses
- Required plan scope: `chatgpt.tokens.use.direct`

The implementation follows OpenAI's Sign in with ChatGPT documentation for open-source/local applications.

