"""Helpers for extracting plain text from common document formats."""

from __future__ import annotations

import importlib
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Callable, Dict


class RagContentExtractor:
    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Path], str]] = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".pptx": self._extract_pptx,
            ".odt": self._extract_odt,
            ".rtf": self._extract_rtf,
            ".eml": self._extract_eml,
            ".msg": self._extract_msg,
            ".html": self._extract_html,
            ".htm": self._extract_html,
        }

    def extract(self, path: Path) -> str:
        handler = self._handlers.get(path.suffix.lower())
        if handler is None:
            return ""
        return handler(path)

    def _extract_pdf(self, path: Path) -> str:
        pdf_module = self._import_optional("pypdf")
        if pdf_module is None:
            return ""
        reader = pdf_module.PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _extract_docx(self, path: Path) -> str:
        docx_module = self._import_optional("docx")
        if docx_module is None:
            return ""
        document = docx_module.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs if p.text)

    def _extract_pptx(self, path: Path) -> str:
        pptx_module = self._import_optional("pptx")
        if pptx_module is None:
            return ""
        presentation = pptx_module.Presentation(str(path))
        parts = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                text = getattr(shape, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _extract_odt(self, path: Path) -> str:
        odf_module = self._import_optional("odf.opendocument")
        text_module = self._import_optional("odf.text")
        if odf_module is None or text_module is None:
            return ""
        document = odf_module.load(str(path))
        parts = []
        for paragraph in document.getElementsByType(text_module.P):
            text = "".join(node.data for node in paragraph.childNodes if node.nodeType == node.TEXT_NODE)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def _extract_rtf(self, path: Path) -> str:
        striprtf_module = self._import_optional("striprtf.striprtf")
        if striprtf_module is None:
            return ""
        data = path.read_text(encoding="utf-8", errors="ignore")
        return striprtf_module.rtf_to_text(data)

    def _extract_eml(self, path: Path) -> str:
        with path.open("rb") as handle:
            message = BytesParser(policy=policy.default).parse(handle)

        parts = []
        for part in message.walk():
            if part.get_content_maintype() != "text":
                continue
            content = part.get_content()
            if part.get_content_subtype() == "html":
                content = self._strip_html(content)
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def _extract_msg(self, path: Path) -> str:
        msg_module = self._import_optional("extract_msg")
        if msg_module is None:
            return ""
        msg = msg_module.Message(str(path))
        msg.process()
        parts = [msg.subject or "", msg.body or ""]
        return "\n\n".join(part for part in parts if part)

    def _extract_html(self, path: Path) -> str:
        data = path.read_text(encoding="utf-8", errors="ignore")
        return self._strip_html(data)

    def _strip_html(self, data: str) -> str:
        bs4_module = self._import_optional("bs4")
        if bs4_module is None:
            return ""
        soup = bs4_module.BeautifulSoup(data, "html.parser")
        return soup.get_text(separator="\n")

    def _import_optional(self, module_name: str):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return None
