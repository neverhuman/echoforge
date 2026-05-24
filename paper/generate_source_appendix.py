#!/usr/bin/env python3
"""Generate the one-column source appendix for the EchoForge paper."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "paper" / "source_appendix_manifest.json"
DEFAULT_OUT = REPO_ROOT / "target" / "paper" / "source_appendix" / "source_code_appendix.tex"
DEFAULT_META = REPO_ROOT / "target" / "paper" / "source_appendix" / "source_appendix_metadata.json"
DEFAULT_ESCROW = REPO_ROOT / "target" / "paper" / "source_appendix" / "ip_escrow_manifest.json"


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _symbol_spans(path: Path) -> dict[str, tuple[int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            spans[node.name] = (int(node.lineno), int(getattr(node, "end_lineno", node.lineno)))
    return spans


def _origin_macro(origin: str) -> str:
    if origin == "generated-evolved-origin":
        return "SrcLineGen"
    if origin == "mixed-origin":
        return "SrcLineMixed"
    if origin == "redacted-escrow-only":
        return "SrcLineIP"
    return "SrcLineHuman"


def _extract_symbol(path: Path, symbol: str) -> tuple[int, int, str]:
    spans = _symbol_spans(path)
    if symbol not in spans:
        raise KeyError(f"{path}: missing symbol {symbol}")
    start, end = spans[symbol]
    lines = path.read_text(encoding="utf-8").splitlines()
    return start, end, "\n".join(lines[start - 1 : end])


def _emit_listing_block(block: dict[str, Any], *, strict: bool) -> tuple[str, list[dict[str, Any]]]:
    rel_path = str(block["path"])
    path = REPO_ROOT / rel_path
    origin = str(block.get("origin", "human-origin"))
    macro = _origin_macro(origin)
    title = str(block.get("title", block.get("id", rel_path)))
    if not path.is_file():
        if strict:
            raise FileNotFoundError(path)
        return (
            f"\\subsection{{{_tex_escape(title)}}}\nMissing source file: \\texttt{{{_tex_escape(rel_path)}}}.\n",
            [],
        )
    chunks = [
        f"\\subsection{{{_tex_escape(title)}}}",
        f"\\noindent\\textit{{Source:}} \\texttt{{{_tex_escape(rel_path)}}}; \\textit{{origin:}} {_tex_escape(origin)}.\\par",
    ]
    metadata_rows: list[dict[str, Any]] = []
    for symbol in [str(item) for item in block.get("symbols", [])]:
        start, end, excerpt = _extract_symbol(path, symbol)
        digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        chunks.append(f"\\subsubsection*{{{_tex_escape(symbol)} (lines {start}--{end})}}")
        for offset, line in enumerate(excerpt.splitlines(), start=start):
            chunks.append(f"\\{macro}{{{_tex_escape(f'{offset:04d}  {line}')}}}")
        metadata_rows.append(
            {
                "block_id": block.get("id", ""),
                "title": title,
                "path": rel_path,
                "origin": origin,
                "symbol": symbol,
                "line_start": start,
                "line_end": end,
                "sha256": digest,
            }
        )
    return "\n".join(chunks) + "\n", metadata_rows


def _xor_stream(payload: bytes, key: bytes) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < len(payload):
        stream.extend(hashlib.sha256(key + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(left ^ right for left, right in zip(payload, stream[: len(payload)]))


def _escrow_payload(manifest: dict[str, Any], *, strict: bool) -> tuple[str, dict[str, Any]]:
    spec = manifest.get("ip_escrow", {})
    if not isinstance(spec, dict) or not spec:
        return "", {}
    rel_path = str(spec.get("path", ""))
    path = REPO_ROOT / rel_path
    excerpts: list[str] = []
    rows: list[dict[str, Any]] = []
    for symbol in [str(item) for item in spec.get("symbols", [])]:
        start, end, excerpt = _extract_symbol(path, symbol)
        excerpts.append(f"# {rel_path}:{start}-{end} {symbol}\n{excerpt}")
        rows.append({"symbol": symbol, "line_start": start, "line_end": end})
    plaintext = "\n\n".join(excerpts).encode("utf-8")
    digest = hashlib.sha256(plaintext).hexdigest()
    env_var = str(spec.get("encrypt_env_var", "ECHOFORGE_IP_APPENDIX_KEY"))
    key_text = os.environ.get(env_var, "")
    envelope: dict[str, Any] = {
        "id": spec.get("id", "ip_escrow"),
        "path": rel_path,
        "symbols": rows,
        "plaintext_sha256": digest,
        "claim_boundary": spec.get("claim_boundary", ""),
    }
    if key_text:
        key = hashlib.sha256(key_text.encode("utf-8")).digest()
        ciphertext = _xor_stream(plaintext, key)
        envelope.update(
            {
                "status": "encrypted_with_env_key",
                "cipher": "sha256-stream-xor-with-hmac-review-envelope",
                "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
                "hmac_sha256_b64": base64.b64encode(
                    hmac.new(key, ciphertext, hashlib.sha256).digest()
                ).decode("ascii"),
            }
        )
    else:
        envelope.update(
            {
                "status": "hash_only_no_key",
                "ciphertext_b64": "",
                "note": f"Set {env_var} to emit an escrow-encrypted duplicate.",
            }
        )
        if strict and not digest:
            raise ValueError("failed to create escrow digest")
    tex = "\n".join(
        [
            "\\subsection{IP Escrow and Redaction Boundary}",
            "The escrow excerpt is a duplicate review artifact. It is not required to reproduce the paper KPI; claim-critical code remains visible in repository source and in the listings above.\\par",
            f"\\SrcLineIP{{{_tex_escape('escrow digest ' + digest)}}}",
            f"\\SrcLineIP{{{_tex_escape('metadata ' + str(DEFAULT_ESCROW.relative_to(REPO_ROOT)))}}}",
        ]
    )
    return tex + "\n", envelope


def build(
    manifest_path: Path, out_path: Path, metadata_path: Path, escrow_path: Path, *, strict: bool
) -> None:
    manifest = _read_json(manifest_path)
    chunks = [
        "% Generated by paper/generate_source_appendix.py; do not edit by hand.",
        "\\section{Source Provenance Appendix}",
        "The listings below are generated from \\texttt{paper/source\\_appendix\\_manifest.json}. Color key: \\SrcKeyHuman{}, \\SrcKeyGen{}, \\SrcKeyMixed{}, and \\SrcKeyIP{}.\\par",
    ]
    metadata_rows: list[dict[str, Any]] = []
    for block in manifest.get("blocks", []):
        if not isinstance(block, dict):
            continue
        tex, rows = _emit_listing_block(block, strict=strict)
        chunks.append(tex)
        metadata_rows.extend(rows)
    escrow_tex, escrow = _escrow_payload(manifest, strict=strict)
    if escrow_tex:
        chunks.append(escrow_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    escrow_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "version": manifest.get("version"),
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "generated_tex": str(out_path.relative_to(REPO_ROOT)),
                "listings": metadata_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    escrow_path.write_text(json.dumps(escrow, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {escrow_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--escrow", type=Path, default=DEFAULT_ESCROW)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    build(args.manifest, args.out, args.metadata, args.escrow, strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
