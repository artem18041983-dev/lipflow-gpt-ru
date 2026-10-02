import base64
import io

import numpy as np
from PIL import Image

from lipflow.chatgpt_vision import make_contact_sheets, _prompt


def test_contact_sheets_are_png_and_bounded():
    frames = np.arange(30 * 96 * 96, dtype=np.uint8).reshape(30, 96, 96)
    sheets = make_contact_sheets(frames, max_frames=24)
    assert len(sheets) == 2
    for url in sheets:
        assert url.startswith("data:image/png;base64,")
        raw = base64.b64decode(url.split(",", 1)[1])
        image = Image.open(io.BytesIO(raw))
        assert image.format == "PNG"
        assert image.width == 576
        assert image.height <= 504


def test_prompt_preserves_language_and_context():
    text = _prompt("ru", "Предыдущая фраза", ["SANEG", "ERIELL"])
    assert "Cyrillic Russian" in text
    assert "Предыдущая фраза" in text
    assert "SANEG, ERIELL" in text
    assert "[unclear]" in text

