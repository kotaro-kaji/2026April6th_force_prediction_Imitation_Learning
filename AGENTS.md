# LLM Wiki

A domain-agnostic pattern for building and maintaining a personal or team knowledge base with an LLM.

## Purpose

This file defines how an LLM should operate as a wiki maintainer, not as a one-off Q&A bot.
The goal is to accumulate knowledge over time in markdown files, with clear links, updates, and revision history.

## Core Model

The system has three layers:

1. **Raw sources** (`raw/`): immutable input material (articles, notes, docs, transcripts, images, data).
2. **Wiki pages** (`wiki/`): maintained markdown knowledge base generated and updated by the LLM.
3. **Schema** (`AGENTS.md`): operating rules, page conventions, and maintenance workflows.

Raw sources are never edited. Wiki pages are continuously refined.

## Standard Operations

### Ingest

When a new source is added:

1. Read the source.
2. Extract key claims, entities, concepts, and evidence.
3. Create or update related wiki pages.
4. Update `wiki/index.md`.
5. Append an entry to `wiki/log.md`.

### Query

When answering questions:

1. Read `wiki/index.md` first.
2. Open relevant pages.
3. Synthesize an answer with page citations.
4. If the output is reusable, save it as a new wiki page and link it from index.

### Lint

Periodically check wiki health:

- Contradictions between pages
- Stale claims superseded by newer sources
- Orphan pages (no inbound links)
- Missing concept/entity pages
- Broken or weak cross-references

## Index and Log Conventions

- `wiki/index.md`: categorized catalog of pages with one-line summaries.
- `wiki/log.md`: append-only timeline of operations.
- Log header format: `## [YYYY-MM-DD] <operation> | <title>`

Example:
`## [2026-04-07] ingest | Product Strategy Memo`

## Authoring Rules

- Prefer short, explicit markdown sections.
- Use descriptive headings and stable filenames.
- Preserve traceability: when possible, note source page/file used for claims.
- Favor incremental edits over large rewrites.
- Keep the wiki broadly useful across domains (personal, business, learning, technical, creative, etc.).
- Add web material to `raw/` only when it is clearly useful for future work and likely worth preserving locally.
- Avoid filling `raw/` with easily re-findable material; prefer keeping such content as citations or summaries in `wiki/`.
- `raw/` is appropriate when re-accessing the source later would be difficult, unstable, or costly in time.

## Optional Tooling

At small scale, index-based navigation is enough. At larger scale, add search tools (CLI or MCP) for faster retrieval.

## Reporting Protocol

- To reduce chat overhead, all progress reports and completion reports must be written to files under `outputs/` instead of conversational updates.
- Default report file: `outputs/report.md` (append updates chronologically).
- When needed, create task-specific reports such as `outputs/report_<task_name>.md`.
- Each report entry should include: timestamp, operation type (`ingest` / `query` / `lint` / `maintenance`), changed files, and a short result summary.
