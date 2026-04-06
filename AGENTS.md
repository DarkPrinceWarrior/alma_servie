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
