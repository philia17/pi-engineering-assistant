"""RAG pipeline for answering EEE queries using local LLM and ChromaDB."""

import os
import re
import warnings
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from langchain_classic.chains import RetrievalQA  # type: ignore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # type: ignore[import-untyped]
from langchain_core.prompts import PromptTemplate  # type: ignore[import-untyped]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

from src.llm_client import generate as ollama_generate
from src.llm_client import strip_thinking

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("EEE_RAG_MODEL", "qwen3:1.7b")
DEFAULT_TOP_K = int(os.getenv("EEE_RAG_TOP_K", "5"))
NUM_PREDICT = int(os.getenv("EEE_RAG_NUM_PREDICT", "512"))
NUM_CTX = int(os.getenv("EEE_RAG_NUM_CTX", "2048"))
# This matches web_ui.vision_fusion.MAX_AUGMENTED_QUERY_CHARACTERS
# (4,000-character question + 2,000-character context + two newlines).  Keep the
# value here to avoid making the core RAG pipeline depend on the web UI package.
MAX_RETRIEVAL_QUERY_CHARACTERS: int = 6_002

_FRONTMATTER_DELIMITER = "---"
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CONTEXT_LABEL_LINE = re.compile(r"(?im)^\s*(?:source note|section):.*$")
REQUIRED_NOTE_METADATA: tuple[str, ...] = (
    "subject",
    "module",
    "topic",
    "source_type",
    "owner",
    "last_updated",
    "review_status",
    "canonical_for",
    "aliases",
    "tags",
)


def _build_ollama_llm() -> Ollama:
    """Create an Ollama LLM for retrieval-only chain wiring."""
    return Ollama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
    )


ANSWER_INPUT_REQUIREMENTS: str = """Use problem inputs only from the current user question or
earlier user-provided conversation facts. Prior assistant answers are not verified
problem inputs. Study-note examples provide methods, not the user's numerical values;
never substitute example numbers into the user's problem.
For numerical tasks, state the applicable formula and its conditions, substitute the
supplied values, and calculate every requested result with units where applicable.
A formula alone is not a completed calculation. If a required value or a matrix,
circuit or signal definition is missing, ask only for the missing input; do not assume it.
For conceptual questions, give a direct explanation without unnecessary formulas.
"""

PROMPT_TEMPLATE: str = (
    """Use the following EEE notes for grounded methods and factual claims.
Answer the requested question directly, not only the name or location of a note.
Treat corrections in "Common Misconceptions" and "Do not confuse with" sections as
the facts to state; never present their incorrect alternatives as true.
Do not include file paths, placeholders, or a repeated "Answer:" label in the response.
If the notes do not support the required method or facts, say "I don't know."
"""
    + ANSWER_INPUT_REQUIREMENTS
    + """

Context:
{context}

Question: {question}
Answer:"""
)

RAG_SYSTEM_PROMPT: str = (
    """Ground methods and factual claims only in the supplied study-note context.
Give a direct, concise factual answer. Include a requested formula or condition rather
than merely naming a note. When asked which note to use, identify the source note
explicitly shown in the context; do not guess another filename. Treat the correction
column in misconception tables as factual, never the incorrect statement. Do not
output file paths, placeholders, or an Answer: label.
"""
    + ANSWER_INPUT_REQUIREMENTS
)


def _parse_frontmatter(content: str) -> tuple[str, dict[str, str]]:
    """Remove simple YAML frontmatter and return Chroma-safe string metadata."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        return content, {}

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONTMATTER_DELIMITER
        )
    except StopIteration:
        return content, {}

    metadata: dict[str, str] = {}
    current_list_key: str | None = None
    list_values: dict[str, list[str]] = {}
    for line in lines[1:closing_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_list_key:
            list_values.setdefault(current_list_key, []).append(stripped[1:].strip().strip("\"'"))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        current_list_key = key if not value else None
        if value:
            metadata[key] = value

    for key, values in list_values.items():
        metadata[key] = ", ".join(value for value in values if value)

    body = "\n".join(lines[closing_index + 1 :]).lstrip("\n")
    if content.endswith("\n"):
        body += "\n"
    return body, metadata


def load_documents(data_dir: Path = DATA_DIR, glob: str = "**/*.md") -> list[Document]:
    """Load only formal Markdown notes with complete required frontmatter."""
    loader = DirectoryLoader(
        str(data_dir),
        glob=glob,
        loader_cls=TextLoader,
    )
    docs = loader.load()
    # Normalize source paths to be relative to repo root
    repo_root = data_dir.parent.parent.resolve()
    knowledge_docs: list[Document] = []
    for doc in docs:
        doc.page_content, frontmatter = _parse_frontmatter(doc.page_content)
        if not all(frontmatter.get(field, "").strip() for field in REQUIRED_NOTE_METADATA):
            continue
        doc.metadata.update(frontmatter)
        source = doc.metadata.get("source", "")
        if source:
            abs_source = Path(source).resolve()
            with suppress(ValueError):
                doc.metadata["source"] = str(abs_source.relative_to(repo_root))
        knowledge_docs.append(doc)
    return knowledge_docs


def _split_markdown_sections(doc: Document) -> list[tuple[str, Document]]:
    """Split a Markdown document into sections with their full heading paths."""
    sections: list[tuple[str, Document]] = []
    headings: list[tuple[int, str]] = []
    body_lines: list[str] = []
    in_fence = False

    def append_section() -> None:
        body = "\n".join(body_lines).strip()
        if not body:
            return
        prefix = "\n".join(f"{'#' * level} {heading}" for level, heading in headings)
        metadata = dict(doc.metadata)
        if headings:
            metadata["heading_path"] = " > ".join(heading for _, heading in headings)
        sections.append((prefix, Document(page_content=body, metadata=metadata)))

    for line in doc.page_content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        heading_match = None if in_fence else _HEADING_PATTERN.match(line)
        if heading_match:
            append_section()
            body_lines = []
            level = len(heading_match.group(1))
            headings = [item for item in headings if item[0] < level]
            headings.append((level, heading_match.group(2).strip()))
        else:
            body_lines.append(line)
    append_section()
    return sections


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split Markdown by section, then by characters while retaining headings.

    ``chunk_size`` and ``chunk_overlap`` are character counts, matching the
    historical ``EEE_RAG_CHUNK_SIZE`` and ``EEE_RAG_CHUNK_OVERLAP`` behavior.
    Blank-line-first splitting keeps most tables, display formulas, and fenced
    ASCII diagrams intact when they fit within a chunk.
    """
    chunks: list[Document] = []
    for doc in docs:
        sections = _split_markdown_sections(doc)
        if not sections and doc.page_content.strip():
            sections = [("", doc)]
        for heading_prefix, section in sections:
            prefix = f"{heading_prefix}\n\n" if heading_prefix else ""
            available_body_size = chunk_size - len(prefix)
            body_chunk_size = available_body_size if available_body_size > 0 else chunk_size
            body_overlap = min(chunk_overlap, max(0, body_chunk_size - 1))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=body_chunk_size,
                chunk_overlap=body_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for body_chunk in splitter.split_documents([section]):
                body_chunk.page_content = f"{prefix}{body_chunk.page_content}"
                chunks.append(body_chunk)
    return chunks


def format_context(documents: Iterable[Document]) -> str:
    """Label retrieved chunks with their source and heading for grounded answers."""
    blocks: list[str] = []
    for document in documents:
        source = str(document.metadata.get("source", "unknown source"))
        heading_path = str(document.metadata.get("heading_path", ""))
        labels = [f"Source note: {source}"]
        if heading_path:
            labels.append(f"Section: {heading_path}")
        labels.append(document.page_content)
        blocks.append("\n".join(labels))
    return "\n\n".join(blocks)


def unique_sources(documents: Iterable[Document]) -> list[str]:
    """Return retrieved source paths once each, preserving retrieval order."""
    return list(dict.fromkeys(str(document.metadata.get("source", "")) for document in documents))


def retrieve_documents(
    question: str,
    chain: Any,
    *,
    vectorstore: Any | None = None,
    rerank: bool = False,
    candidates: int | None = None,
    keep: int | None = None,
) -> list[Document]:
    """Retrieve chunks for one question, optionally reranking inside one subject store."""
    keep_k = keep if keep is not None else DEFAULT_TOP_K
    if rerank:
        store = vectorstore
        if store is None:
            store = getattr(chain.retriever, "vectorstore", None)
        if store is None:
            raise ValueError("rerank requires a vectorstore on the chain or an explicit store")
        candidate_k = candidates if candidates is not None else max(keep_k * 4, 20)
        docs = store.similarity_search(question, k=candidate_k)
        from rag.retrievers.rerank import rerank_chunks

        return rerank_chunks(question, docs, keep=keep_k)
    docs = chain.retriever.invoke(question)
    return list(docs)


def strip_context_labels(answer: str) -> str:
    """Remove source and heading labels that a model copied from RAG context."""
    without_labels = _CONTEXT_LABEL_LINE.sub("", answer)
    return re.sub(r"\n{3,}", "\n\n", without_labels).strip()


def configure_embedding_runtime(*, threads: int | None = None) -> int:
    """Pin embedding/BLAS thread counts so Pi builds cannot starve SSH."""
    configured = threads
    if configured is None:
        configured = int(os.getenv("EEE_EMBED_THREADS", "1"))
    thread_count = max(1, configured)
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["EEE_EMBED_THREADS"] = str(thread_count)
    return thread_count


def build_vectorstore(docs: Any, persist_directory: Path = CHROMA_DIR) -> Any:
    """Chunk documents by character count and store embeddings in ChromaDB."""
    configure_embedding_runtime()
    chunk_size = int(os.getenv("EEE_RAG_CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("EEE_RAG_CHUNK_OVERLAP", "64"))
    chunks = split_documents(docs, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"local_files_only": True},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    return vectorstore


def load_vectorstore(persist_directory: Path = CHROMA_DIR) -> Any:
    """Load an existing ChromaDB vectorstore without opening a frozen v2 index."""
    from rag.mutable_index_runtime import mutable_index_copy

    mutable_copy = mutable_index_copy(
        persist_directory,
        release_root=Path(__file__).resolve().parent.parent,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"local_files_only": True},
    )
    return Chroma(
        persist_directory=str(mutable_copy.mutable_index),
        embedding_function=embeddings,
    )


def build_rag_chain(vectorstore: Any, k: int = DEFAULT_TOP_K) -> Any:
    """Build a RetrievalQA chain backed by Ollama."""
    llm = _build_ollama_llm()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def normalize_retrieval_query(retrieval_query: object | None) -> str | None:
    """Return a safe bounded retrieval hint or ``None`` for text-only fallback.

    Optional visual retrieval hints are request-local and untrusted.  The normal
    RAG path must keep working when a caller supplies an invalid hint, so callers
    use ``None`` to retrieve with their unchanged original question instead.
    Newlines are permitted for the bounded fusion format; every other
    non-printable character is rejected.
    """
    if not isinstance(retrieval_query, str):
        return None
    if not retrieval_query.strip() or len(retrieval_query) > MAX_RETRIEVAL_QUERY_CHARACTERS:
        return None
    if any(character != "\n" and not character.isprintable() for character in retrieval_query):
        return None
    return retrieval_query


def query(
    question: str,
    chain: Any,
    *,
    model: str | None = None,
    vectorstore: Any | None = None,
    rerank: bool | None = None,
    candidates: int | None = None,
    keep: int | None = None,
    retrieval_query: object | None = None,
) -> dict[str, Any]:
    """Retrieve with an optional safe hint while answering the original question."""
    use_rerank = (
        rerank
        if rerank is not None
        else os.getenv("EEE_RAG_RERANK", "1").strip() not in {"0", "false", "False"}
    )
    retrieval_input = normalize_retrieval_query(retrieval_query) or question
    docs = retrieve_documents(
        retrieval_input,
        chain,
        vectorstore=vectorstore,
        rerank=use_rerank,
        candidates=candidates,
        keep=keep,
    )
    sources = unique_sources(docs)
    context = format_context(docs)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = ollama_generate(
        prompt,
        model=model or LLM_MODEL,
        system_prompt=RAG_SYSTEM_PROMPT,
        think=False,
        timeout=300,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
    )
    thinking = result["thinking"]
    answer = result["answer"]
    extra_thinking, answer = strip_thinking(answer)
    if extra_thinking:
        thinking = f"{thinking}\n{extra_thinking}".strip() if thinking else extra_thinking
    answer = strip_context_labels(answer)
    return {
        "answer": answer,
        "thinking": thinking,
        "sources": sources,
        "documents": docs,
        "reranked": use_rerank,
    }
