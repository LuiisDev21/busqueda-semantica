#!/usr/bin/env python3
"""
main.py -- Buscador semantico conceptual con embeddings vectoriales.

Uso:
    python main.py                     # modo interactivo
    python main.py "consulta"          # busqueda directa

Comandos interactivos:
    <consulta>       busqueda semantica
    /c <consulta>    comparar semantico vs TF-IDF
    /top <N>         cambiar cantidad de resultados
    /build           reconstruir indice
    /help            mostrar comandos
    /q               salir
"""

import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from build_index import cargar_corpus, generar_embeddings, guardar_indice, CorpusError

# -- Constantes ---------------------------------------------------------------

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
INDICE_NPY = "index/embeddings_index.npy"
METADATA_JSON = "index/docs_metadata.json"
CORPUS_DIR = "corpus"

console = Console()


# -- Indice -------------------------------------------------------------------

def cargar_indice() -> tuple[np.ndarray, list[dict]]:
    """Carga embeddings y metadatos desde disco."""
    if not os.path.exists(INDICE_NPY):
        console.print(f"\n[red]Error:[/] No se encontro '[bold]{INDICE_NPY}[/]'. Ejecuta [bold]/build[/] para generar el indice.")
        return None, None
    if not os.path.exists(METADATA_JSON):
        console.print(f"\n[red]Error:[/] No se encontro '[bold]{METADATA_JSON}[/]'. Ejecuta [bold]/build[/] para generar el indice.")
        return None, None

    embeddings = np.load(INDICE_NPY)
    with open(METADATA_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return embeddings, docs


def construir_indice():
    """Reconstruye el indice desde el corpus."""
    try:
        docs = cargar_corpus(CORPUS_DIR)
        embeddings = generar_embeddings(docs, MODELO)
        guardar_indice(embeddings, docs, "index")
        return embeddings, docs
    except CorpusError as e:
        console.print(f"[red]Error:[/] {e}")
        return None, None


# -- Busqueda -----------------------------------------------------------------

def _ranquear(docs: list[dict], similitudes: np.ndarray, top_k: int) -> list[dict]:
    """Builder de resultados: ordena por score y empaqueta top_k documentos.

    ponytail: unifica el armado de resultados que antes estaba duplicado
    en buscar_semantico() y buscar_tfidf().
    """
    indices = np.argsort(similitudes)[::-1][:top_k]
    return [
        {
            "id": docs[i]["id"],
            "titulo": docs[i]["titulo"],
            "archivo": docs[i]["archivo"],
            "score": float(similitudes[i]),
            "texto": docs[i]["texto"],
        }
        for i in indices
    ]


def buscar_semantico(query: str, model, corpus_embeddings: np.ndarray, docs: list[dict], top_k: int) -> list[dict]:
    query_vec = model.encode([query])
    similitudes = cosine_similarity(query_vec, corpus_embeddings)[0]
    return _ranquear(docs, similitudes, top_k)


def buscar_tfidf(query: str, docs: list[dict], top_k: int) -> list[dict]:
    textos = [d["texto"] for d in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    query_vec = vectorizer.transform([query])
    similitudes = cosine_similarity(query_vec, tfidf_matrix)[0]
    return _ranquear(docs, similitudes, top_k)


# -- UI -----------------------------------------------------------------------

def _barra(score: float) -> str:
    pct = score * 100
    n = min(int(pct / 5), 20)
    color = "green" if pct > 70 else "yellow" if pct > 40 else "red"
    return f"[{color}]{'#' * n}{'-' * (20 - n)}[/]"


def mostrar_resultados(query: str, resultados: list[dict], metodo: str):
    console.print()
    console.print(Panel(
        f"[bold]Query:[/] {query}\n[dim]Metodo: {metodo}[/]",
        title="Busqueda Semantica",
        border_style="cyan",
        box=box.HEAVY,
        padding=(1, 2),
    ))

    if not resultados:
        console.print("  [dim](sin resultados)[/]")
        return

    for i, r in enumerate(resultados, 1):
        pct = r["score"] * 100
        snippet = r["texto"][:200].replace("\n", " ") + "..."

        console.print(f"\n  [bold white]#{i}[/]  [{pct:5.1f}%] {_barra(r['score'])}")
        console.print(f"      [bold]{r['titulo']}[/]")
        console.print(f"      [dim]{os.path.basename(r['archivo'])}[/]")
        console.print(f"      [dim]{'-' * 50}[/]")
        console.print(f"      [dim]{snippet}[/]")


def comparar_metodos(query: str, model, corpus_embeddings, docs, top_k: int):
    sem = buscar_semantico(query, model, corpus_embeddings, docs, top_k)
    tfidf = buscar_tfidf(query, docs, top_k)

    console.print()
    console.print(Panel(
        f"[bold]Query:[/] {query}",
        title="Comparativa: Semantico vs TF-IDF",
        border_style="cyan",
        box=box.HEAVY,
        padding=(1, 2),
    ))

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("Busqueda Semantica (embeddings)", style="green")
    table.add_column("", style="dim", width=1)
    table.add_column("Baseline TF-IDF (lexical)", style="yellow")

    for i in range(top_k):
        s = sem[i] if i < len(sem) else None
        t = tfidf[i] if i < len(tfidf) else None
        izq = f"{s['titulo']}  [{s['score']*100:.1f}%]" if s else "-"
        der = f"{t['titulo']}  [{t['score']*100:.1f}%]" if t else "-"
        table.add_row(str(i + 1), izq, "", der)

    console.print(table)

    ids_sem = {r["id"] for r in sem}
    ids_tfidf = {r["id"] for r in tfidf}
    solapados = ids_sem & ids_tfidf

    console.print()
    console.print(Panel(
        f"Solapamiento en top-{top_k}: [bold]{len(solapados)}/{top_k}[/] documentos en comun\n\n"
        f"[dim]Los embeddings capturan [bold]significado[/]; TF-IDF captura [bold]palabras exactas[/].[/]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def mostrar_ayuda():
    console.print()
    console.print(Panel(
        "[bold white]Comandos disponibles[/]\n\n"
        "  [bold]<consulta>[/]       Busqueda semantica\n"
        "  [bold]/c <consulta>[/]    Comparar semantico vs TF-IDF\n"
        "  [bold]/top <N>[/]         Cambiar cantidad de resultados (actual: se muestra al inicio)\n"
        "  [bold]/build[/]           Reconstruir el indice desde corpus/\n"
        "  [bold]/help[/]            Mostrar esta ayuda\n"
        "  [bold]/q[/]               Salir\n\n"
        "[dim]Ejemplo: /c como funciona una red neuronal[/]",
        title="Ayuda",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


# -- Modo interactivo ---------------------------------------------------------

def interactivo():
    console.print()
    console.print(Panel(
        "[bold white]BUSCADOR SEMANTICO[/]\n\n"
        "Encuentra documentos por lo que quieren decir, no por las palabras que usan.\n"
        "Escribe [bold]/help[/] para ver los comandos disponibles.",
        border_style="cyan",
        box=box.HEAVY,
        padding=(1, 2),
    ))

    # Cargar indice y modelo
    embeddings, docs = cargar_indice()
    if embeddings is None:
        return

    console.print(f"[dim]Cargando modelo {MODELO}...[/]")
    model = SentenceTransformer(MODELO)
    console.print(f"[dim]Corpus: {len(docs)} documentos | Top-K: 5[/]\n")

    top_k = 5

    while True:
        try:
            entrada = Prompt.ask("  [bold cyan]>[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/]")
            break

        if not entrada:
            continue

        # Comandos
        if entrada == "/q":
            console.print("[dim]Hasta luego.[/]")
            break

        if entrada == "/help":
            mostrar_ayuda()
            continue

        if entrada == "/build":
            console.print("[yellow]Reconstruyendo indice...[/]")
            nuevos = construir_indice()
            if nuevos[0] is not None:
                embeddings, docs = nuevos
                console.print(f"[green]Listo. Corpus: {len(docs)} documentos.[/]")
            continue

        if entrada.startswith("/top"):
            try:
                nuevo_top = int(entrada.split()[1])
                if nuevo_top < 1:
                    raise ValueError
                top_k = nuevo_top
                console.print(f"[green]Top-K: {top_k}[/]")
            except (IndexError, ValueError):
                console.print("[red]Uso: /top <N>  (ej: /top 3)[/]")
            continue

        if entrada.startswith("/c "):
            query = entrada[3:].strip()
            if query:
                comparar_metodos(query, model, embeddings, docs, top_k)
            continue

        if entrada.startswith("/"):
            console.print(f"[red]Comando desconocido:[/] {entrada.split()[0]}. Escribe [bold]/help[/] para ver los comandos.")
            continue

        # Busqueda normal
        resultados = buscar_semantico(entrada, model, embeddings, docs, top_k)
        mostrar_resultados(entrada, resultados, "Busqueda Semantica (embeddings)")


# -- Entrada directa ----------------------------------------------------------

def busqueda_directa(query: str):
    embeddings, docs = cargar_indice()
    if embeddings is None:
        return

    console.print(f"[dim]Cargando modelo {MODELO}...[/]")
    model = SentenceTransformer(MODELO)

    resultados = buscar_semantico(query, model, embeddings, docs, top_k=5)
    mostrar_resultados(query, resultados, "Busqueda Semantica (embeddings)")


# -- Punto de entrada ---------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        busqueda_directa(" ".join(sys.argv[1:]))
    else:
        interactivo()
