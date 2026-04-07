# Project Context

## Overview
- This repository is the alma_servie project.
- Treat this repository as production-like code: prefer minimal, reversible changes.

## Working Rules
- Before changing code, first inspect relevant files and explain the intended change briefly.
- Prefer targeted edits over broad refactors.
- Do not rename files, move modules, or change public interfaces unless explicitly asked.
- Do not edit secrets, credentials, or deployment configs unless explicitly asked.
- Do not install new dependencies unless explicitly asked.

## Tooling Workflow
- Use fff first for broad repo discovery: file search, pattern search, entry-point discovery, and narrowing the search area.
- Use Serena after that for symbol-level navigation, code understanding, and precise edits.
- Prefer Serena tools for symbol-aware changes and code structure exploration.
- Prefer fff for fast file/pattern discovery across the repository.
- Avoid reading whole source files unless it is clearly necessary.
- When working in Hermes worktree mode, activate Serena for the current worktree before symbolic navigation or edits.

## Git and Safety
- Assume the user may have local work in progress outside the active worktree.
- Prefer working in Hermes worktree mode.
- Before risky edits, explain the risk.
- Prefer small patches that are easy to review.

## Code Quality
- Follow existing project style and patterns.
- Reuse existing utilities and conventions before introducing new abstractions.
- Keep comments minimal and useful.
- When adding code, note where tests should go if you do not add them directly.

## Response Style
- Be direct and practical.
- State assumptions explicitly.
- If uncertain, inspect more files before proposing a change.
