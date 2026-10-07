import re

from app.schemas.diff import DiffHunk, DiffLine, ParsedDiffFile, PullRequestFile

HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff_file(file: PullRequestFile) -> ParsedDiffFile:
    if not file.patch:
        return ParsedDiffFile(
            file=file.filename,
            status=file.status,
            additions=file.additions,
            deletions=file.deletions,
            patch=file.patch,
        )

    hunks: list[DiffHunk] = []
    current_hunk: DiffHunk | None = None
    old_line_number = 0
    new_line_number = 0

    for raw_line in file.patch.splitlines():
        header_match = HUNK_HEADER_PATTERN.match(raw_line)
        if header_match:
            old_start = int(header_match.group(1))
            old_count = int(header_match.group(2) or "1")
            new_start = int(header_match.group(3))
            new_count = int(header_match.group(4) or "1")
            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
            hunks.append(current_hunk)
            old_line_number = old_start
            new_line_number = new_start
            continue

        if current_hunk is None or raw_line.startswith("\\ No newline at end of file"):
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            current_hunk.lines.append(
                DiffLine(
                    old_line_number=None,
                    new_line_number=new_line_number,
                    content=raw_line[1:],
                    change_type="added",
                )
            )
            new_line_number += 1
            continue

        if raw_line.startswith("-") and not raw_line.startswith("---"):
            current_hunk.lines.append(
                DiffLine(
                    old_line_number=old_line_number,
                    new_line_number=None,
                    content=raw_line[1:],
                    change_type="deleted",
                )
            )
            old_line_number += 1
            continue

        content = raw_line[1:] if raw_line.startswith(" ") else raw_line
        current_hunk.lines.append(
            DiffLine(
                old_line_number=old_line_number,
                new_line_number=new_line_number,
                content=content,
                change_type="context",
            )
        )
        old_line_number += 1
        new_line_number += 1

    return ParsedDiffFile(
        file=file.filename,
        status=file.status,
        additions=file.additions,
        deletions=file.deletions,
        changed_lines=_collect_changed_lines(hunks),
        deleted_lines=_collect_deleted_lines(hunks),
        hunks=hunks,
        patch=file.patch,
    )


def _collect_changed_lines(hunks: list[DiffHunk]) -> list[int]:
    return [
        line.new_line_number
        for hunk in hunks
        for line in hunk.lines
        if line.change_type == "added" and line.new_line_number is not None
    ]


def _collect_deleted_lines(hunks: list[DiffHunk]) -> list[int]:
    return [
        line.old_line_number
        for hunk in hunks
        for line in hunk.lines
        if line.change_type == "deleted" and line.old_line_number is not None
    ]