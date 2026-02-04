"""
Descarga la taxonomía IPTC Media Topics desde GitHub y genera
el fichero iptc_categories.json con el árbol jerárquico.

Fuente: https://github.com/TajaKuzman/IPTC-Media-Topic-Classification

Uso:
    python src/build_iptc_tree.py
"""

import json
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/TajaKuzman/"
    "IPTC-Media-Topic-Classification/main/data/iptc_mapping.json"
)
OUTPUT_PATH = Path(__file__).parent / "iptc_categories.json"


def download_source() -> dict:
    print(f"Descargando {SOURCE_URL} ...")
    with urllib.request.urlopen(SOURCE_URL) as resp:
        raw = resp.read().decode("utf-8")
    # El JSON original usa NaN de JavaScript en lugar de null
    raw = raw.replace("NaN", "null")
    return json.loads(raw)


def find_level(entry: dict) -> int:
    for lev in (5, 4, 3, 2, 1):
        if entry.get(f"Level{lev}/NewsCode") is not None:
            return lev
    return 0


def build_tree(data: dict) -> dict:
    # Filtrar entradas retiradas
    active = {k: v for k, v in data.items() if v.get("RetiredDate") is None}

    # Indexar por código propio
    by_code = {}
    for entry in active.values():
        level = find_level(entry)
        own_code = entry.get(f"Level{level}/NewsCode")
        parent_code = entry.get(f"Level{level - 1}/NewsCode") if level > 1 else None
        by_code[own_code] = {
            "name": entry["Name (en-GB)"],
            "level": level,
            "parent_code": parent_code,
        }

    def children_of(parent_code: str, parent_level: int) -> dict | None:
        children = {}
        child_level = parent_level + 1
        for code, info in by_code.items():
            if info["parent_code"] == parent_code and info["level"] == child_level:
                subtree = children_of(code, child_level)
                children[info["name"]] = subtree if subtree else None
        return children if children else None

    # Construir desde las raíces (nivel 1, sin padre)
    tree = {}
    for code, info in sorted(by_code.items()):
        if info["level"] == 1:
            subtree = children_of(code, 1)
            tree[info["name"]] = subtree if subtree else None

    return tree


def count_leaves(node: dict) -> int:
    count = 0
    for children in node.values():
        if children is None:
            count += 1
        else:
            count += count_leaves(children)
    return count


def max_depth(node: dict, depth: int = 1) -> int:
    if node is None:
        return depth
    return max(max_depth(v, depth + 1) for v in node.values())


def main():
    data = download_source()
    tree = build_tree(data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print(f"Generado: {OUTPUT_PATH}")
    print(f"  Categorías raíz: {len(tree)}")
    print(f"  Hojas totales:   {count_leaves(tree)}")
    print(f"  Profundidad máx: {max_depth(tree)}")


if __name__ == "__main__":
    main()
