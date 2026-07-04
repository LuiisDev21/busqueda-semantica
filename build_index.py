#!/usr/bin/env python3
"""
build_index.py -- Indexa un corpus de documentos de texto usando embeddings semanticos.

Uso:
    python build_index.py [--model MODELO] [--corpus DIR]

Genera dos archivos:
    - embeddings_index.npy  : matriz de vectores (N_docs x D)
    - docs_metadata.json    : metadatos de cada documento (id, titulo, ruta, texto completo)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


class CorpusError(Exception):
    """Error al cargar el corpus de documentos."""
    pass


def cargar_corpus(directorio: str) -> list[dict]:
    """Lee todos los archivos .txt del directorio y devuelve lista de documentos.

    Lanza CorpusError si el directorio no existe o esta vacio.
    """
    ruta_dir = Path(directorio)

    if not ruta_dir.is_dir():
        raise CorpusError(f"El directorio '{directorio}' no existe.")

    archivos = sorted(ruta_dir.glob("*.txt"))
    if not archivos:
        raise CorpusError(f"No se encontraron archivos .txt en '{directorio}'.")

    docs = []
    for i, archivo in enumerate(archivos):
        texto = archivo.read_text(encoding="utf-8").strip()
        if texto:
            titulo = texto.split("\n")[0].strip()
            docs.append({
                "id": i,
                "titulo": titulo,
                "archivo": str(archivo),
                "texto": texto,
            })

    console.print(f"  [green]Documentos cargados:[/] {len(docs)} desde '[bold]{directorio}[/]'")
    return docs


def generar_embeddings(docs: list[dict], modelo: str) -> np.ndarray:
    """Genera embeddings para cada documento usando el modelo especificado."""
    console.print(f"\n  [yellow]Cargando modelo:[/] [bold]{modelo}[/]")
    console.print("  [dim](la primera ejecucion descarga ~80-500 MB a la cache de HuggingFace)[/]\n")
    model = SentenceTransformer(modelo)

    textos = [doc["texto"] for doc in docs]
    console.print(f"  [yellow]Generando embeddings[/] para {len(textos)} documentos...")
    embeddings = model.encode(textos, show_progress_bar=True)

    console.print(f"  [green]Embeddings generados:[/] matriz {embeddings.shape} (tipo: {embeddings.dtype})")
    return embeddings


def guardar_indice(embeddings: np.ndarray, docs: list[dict], salida_dir: str):
    """Guarda embeddings en .npy y metadatos en .json."""
    ruta_npy = os.path.join(salida_dir, "embeddings_index.npy")
    ruta_json = os.path.join(salida_dir, "docs_metadata.json")

    np.save(ruta_npy, embeddings)
    size_kb = os.path.getsize(ruta_npy) / 1024
    console.print(f"  [green]Embeddings guardados:[/] [bold]{ruta_npy}[/] ({size_kb:.1f} KB)")

    metadata = [{"id": d["id"], "titulo": d["titulo"], "archivo": d["archivo"], "texto": d["texto"]} for d in docs]

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    console.print(f"  [green]Metadatos guardados:[/] [bold]{ruta_json}[/] ({len(metadata)} documentos)")


def main():
    parser = argparse.ArgumentParser(
        description="Indexa un corpus de documentos usando embeddings semanticos."
    )
    parser.add_argument(
        "--model", default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Modelo de sentence-transformers (default: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser.add_argument(
        "--corpus", default="corpus",
        help="Directorio con archivos .txt (default: corpus/)"
    )
    parser.add_argument(
        "--output", default="index",
        help="Directorio de salida (default: index/)"
    )
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold white]INDEXADOR SEMANTICO[/]\n[dim]Embeddings + Busqueda Conceptual[/]",
        box=box.HEAVY,
        border_style="cyan",
        padding=(1, 2),
    ))

    try:
        docs = cargar_corpus(args.corpus)
    except CorpusError as e:
        console.print(f"\n[red]Error:[/] {e}")
        return

    embeddings = generar_embeddings(docs, args.model)
    guardar_indice(embeddings, docs, args.output)

    table = Table(title="Resumen de Indexacion", box=box.ROUNDED, border_style="cyan")
    table.add_column("Parametro", style="dim", no_wrap=True)
    table.add_column("Valor", style="bold")
    table.add_row("Modelo", args.model)
    table.add_row("Dimensionalidad", str(embeddings.shape[1]))
    table.add_row("Documentos indexados", str(len(docs)))
    console.print()
    console.print(table)
    console.print("\n[green]Indexacion completada.[/] Ejecuta [bold]python main.py[/] para buscar.\n")


if __name__ == "__main__":
    main()
