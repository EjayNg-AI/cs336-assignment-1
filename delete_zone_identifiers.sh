#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--dry-run|-n]"
    echo "Recursively delete files whose names contain ':Zone.Identifier'."
}

dry_run=false

case "${1:-}" in
    "")
        ;;
    --dry-run|-n)
        dry_run=true
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [[ "$#" -gt 1 ]]; then
    usage >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$dry_run" == true ]]; then
    find "$repo_root" -type f -name '*:Zone.Identifier*' -print
else
    find "$repo_root" -type f -name '*:Zone.Identifier*' -print -delete
fi
