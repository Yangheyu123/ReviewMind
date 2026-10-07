import asyncio

from typing import Any
from urllib.parse import quote

import httpx

from app.core.cache import redis_cache
from app.core.config import settings
from app.schemas.github import (
    GitHubBranchRef,
    GitHubPullRequestFile,
    GitHubPullRequestInfo,
    GitHubPullRequestRef,
)

# 缓存 TTL（秒）
_CACHE_TTL_PR_INFO = 300       # 5 分钟
_CACHE_TTL_PR_FILES = 300      # 5 分钟

# files 分页：单页上限与最多页数（50 页 × 100 = 5000 文件封顶）
_FILES_PER_PAGE = 100
_FILES_MAX_PAGES = 50

# 限流重试：Retry-After / x-ratelimit-remaining=0 / 403 body 明示限流（二级限流
# 常返回不带 Retry-After 头的裸 403，提示仅在 body）时触发；最多 4 次，单次等待封顶 90s
_RATE_LIMIT_MAX_RETRIES = 4
_RATE_LIMIT_BACKOFF_CAP_SECONDS = 90.0


class GitHubClientError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubClient:
    def __init__(
        self,
        api_base_url: str = settings.github_api_base_url,
        token: str | None = settings.github_token,
        timeout_seconds: float = settings.github_timeout_seconds,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def fetch_pull_request(self, pr_ref: GitHubPullRequestRef) -> GitHubPullRequestInfo:
        cache_key = f"github:pr:{pr_ref.owner}/{pr_ref.repo}/{pr_ref.pull_number}"
        # 注入了受控传输（测试/代理场景）时旁路缓存读写，避免 mock 数据污染共享 Redis
        use_cache = self.client is None
        if use_cache:
            cached = await redis_cache.get(cache_key)
            if cached is not None:
                return GitHubPullRequestInfo.model_validate(cached)

        payload = await self._get_json(f"/repos/{pr_ref.owner}/{pr_ref.repo}/pulls/{pr_ref.pull_number}")
        info = GitHubPullRequestInfo(
            owner=pr_ref.owner,
            repo=pr_ref.repo,
            pull_number=pr_ref.pull_number,
            title=str(payload.get("title", "")),
            author=str(payload.get("user", {}).get("login", "unknown")),
            state=str(payload.get("state", "unknown")),
            base=_parse_branch_ref(payload.get("base", {})),
            head=_parse_branch_ref(payload.get("head", {})),
            changed_files=int(payload.get("changed_files", 0)),
            additions=int(payload.get("additions", 0)),
            deletions=int(payload.get("deletions", 0)),
            html_url=str(payload.get("html_url", pr_ref.html_url)),
        )
        if use_cache:
            await redis_cache.set(cache_key, info.model_dump(mode="json"), ttl_seconds=_CACHE_TTL_PR_INFO)
        return info

    async def fetch_pull_request_files(self, pr_ref: GitHubPullRequestRef) -> list[GitHubPullRequestFile]:
        cache_key = f"github:files:{pr_ref.owner}/{pr_ref.repo}/{pr_ref.pull_number}"
        use_cache = self.client is None
        if use_cache:
            cached = await redis_cache.get(cache_key)
            if cached is not None and isinstance(cached, list):
                return [GitHubPullRequestFile.model_validate(item) for item in cached]

        payload: list[Any] = []
        for page in range(1, _FILES_MAX_PAGES + 1):
            batch = await self._get_json(
                f"/repos/{pr_ref.owner}/{pr_ref.repo}/pulls/{pr_ref.pull_number}/files",
                params={"per_page": _FILES_PER_PAGE, "page": page},
            )
            if not isinstance(batch, list):
                raise GitHubClientError("GitHub files response is invalid")
            payload.extend(batch)
            # 不足一页说明已到最后一页（GitHub PR files 默认每页 30 条，历史上
            # 不分页导致超过 30 个文件的 PR 只审前 30 个）
            if len(batch) < _FILES_PER_PAGE:
                break

        files = [
            GitHubPullRequestFile(
                filename=str(item.get("filename", "")),
                status=str(item.get("status", "modified")),
                additions=int(item.get("additions", 0)),
                deletions=int(item.get("deletions", 0)),
                patch=item.get("patch"),
            )
            for item in payload
        ]
        if use_cache:
            await redis_cache.set(cache_key, [f.model_dump(mode="json") for f in files], ttl_seconds=_CACHE_TTL_PR_FILES)
        return files

    async def download_tarball(self, pr_ref: GitHubPullRequestRef, ref: str) -> bytes | None:
        """下载仓库指定 ref 的 tarball（源码快照用）。GitHub 会 302 到 codeload，需跟随重定向。"""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        path = f"/repos/{pr_ref.owner}/{pr_ref.repo}/tarball/{ref}"
        try:
            if self.client is not None:
                response = await self.client.get(path, headers=headers)
            else:
                async with httpx.AsyncClient(
                    base_url=self.api_base_url, timeout=self.timeout_seconds, follow_redirects=True,
                ) as client:
                    response = await client.get(path, headers=headers)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return response.content

    async def fetch_file_content(self, pr_ref: GitHubPullRequestRef, path: str, ref: str) -> str | None:
        """拉取指定 commit ref 下的文件原文（AST 上下文需要完整源码，patch 重建不可靠）。"""
        headers = {
            "Accept": "application/vnd.github.raw",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        quoted_path = quote(path, safe="/")
        url_path = f"/repos/{pr_ref.owner}/{pr_ref.repo}/contents/{quoted_path}"
        try:
            if self.client is not None:
                response = await self.client.get(url_path, headers=headers, params={"ref": ref})
            else:
                async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout_seconds) as client:
                    response = await client.get(url_path, headers=headers, params={"ref": ref})
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return response.text

    async def merge_pull_request(
        self,
        pr_ref: GitHubPullRequestRef,
        commit_title: str | None = None,
        commit_message: str | None = None,
        merge_method: str = "merge",
    ) -> dict[str, Any]:
        """调用 GitHub API 合并 PR。

        PUT /repos/{owner}/{repo}/pulls/{pull_number}/merge

        Args:
            pr_ref: PR 引用
            commit_title: 合并 commit 标题
            commit_message: 合并 commit 消息
            merge_method: 合并方式 — 'merge', 'squash', 或 'rebase'

        Returns:
            {"merged": bool, "message": str, "sha": str | None}
        """
        path = f"/repos/{pr_ref.owner}/{pr_ref.repo}/pulls/{pr_ref.pull_number}/merge"
        body: dict[str, Any] = {"merge_method": merge_method}
        if commit_title:
            body["commit_title"] = commit_title
        if commit_message:
            body["commit_message"] = commit_message

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self.client is not None:
            response = await self.client.put(path, headers=headers, json=body)
            return _handle_merge_response(response)

        async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout_seconds) as client:
            response = await client.put(path, headers=headers, json=body)
            return _handle_merge_response(response)

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        retries = 0
        while True:
            try:
                if self.client is not None:
                    response = await self.client.get(path, headers=headers, params=params)
                else:
                    async with httpx.AsyncClient(base_url=self.api_base_url, timeout=self.timeout_seconds) as client:
                        response = await client.get(path, headers=headers, params=params)
            except httpx.TransportError:
                # 网络层间歇性失败（连接被断/服务器无响应断连/读超时——本机链路
                # 劣化时约 50% 连接失败，且形态多变，故按 TransportError 全家族重试）
                if retries >= _RATE_LIMIT_MAX_RETRIES + 2:
                    raise
                await asyncio.sleep(min(1.0 * (2 ** retries), 15.0))
                retries += 1
                continue
            if response.status_code in (403, 429) and self._is_rate_limited(response):
                if retries >= _RATE_LIMIT_MAX_RETRIES:
                    break
                wait = self._rate_limit_wait_seconds(response, attempt=retries)
                await asyncio.sleep(wait)
                retries += 1
                continue
            # 链路劣化时响应体可能被截断（json 解析失败）——同样按可重试处理
            try:
                response.json()
            except ValueError:
                if retries >= _RATE_LIMIT_MAX_RETRIES + 2:
                    break
                await asyncio.sleep(min(1.0 * (2 ** retries), 15.0))
                retries += 1
                continue
            break
        return _handle_response(response)

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        """仅当响应明确指示限流时才重试（普通 403 权限拒绝立即抛错）。

        GitHub 二级限流（abuse detection）常返回不带 Retry-After 头的裸 403，
        提示只出现在 body（"secondary rate limit" / "rate limit"字样）——
        仅头字段判定会把这类 403 误判为权限拒绝而直接失败。
        """
        if "retry-after" in response.headers:
            return True
        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining == "0":
            return True
        if response.status_code == 403:
            body = (response.text or "").lower()
            return "rate limit" in body
        return False

    @staticmethod
    def _rate_limit_wait_seconds(response: httpx.Response, attempt: int) -> float:
        """优先服务端指示：Retry-After 秒数 / x-ratelimit-reset 距今秒数；否则指数退避。"""
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), _RATE_LIMIT_BACKOFF_CAP_SECONDS)
            except ValueError:
                pass
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                import time
                return min(max(float(reset) - time.time(), 0.0), _RATE_LIMIT_BACKOFF_CAP_SECONDS)
            except ValueError:
                pass
        return min(0.5 * (2 ** attempt), _RATE_LIMIT_BACKOFF_CAP_SECONDS)


def _handle_response(response: httpx.Response) -> Any:
    if response.status_code == 401:
        raise GitHubClientError("GitHub token is invalid", status_code=401)
    if response.status_code == 403:
        raise GitHubClientError("GitHub API rate limit or permission denied", status_code=403)
    if response.status_code == 404:
        raise GitHubClientError("GitHub pull request was not found", status_code=404)
    if response.status_code >= 400:
        raise GitHubClientError("GitHub API request failed", status_code=response.status_code)
    return response.json()


def _handle_merge_response(response: httpx.Response) -> dict[str, Any]:
    """处理 GitHub merge API 响应，包括特殊状态码。"""
    if response.status_code == 200:
        data = response.json()
        return {
            "merged": bool(data.get("merged", False)),
            "message": str(data.get("message", "Merged")),
            "sha": data.get("sha"),
        }
    if response.status_code == 204:
        return {"merged": True, "message": "Already merged", "sha": None}
    if response.status_code == 401:
        raise GitHubClientError("GitHub token is invalid", status_code=401)
    if response.status_code == 403:
        raise GitHubClientError("GitHub API rate limit or permission denied", status_code=403)
    if response.status_code == 404:
        raise GitHubClientError("GitHub pull request was not found", status_code=404)
    if response.status_code == 405:
        raise GitHubClientError("Merge method not allowed for this PR", status_code=405)
    if response.status_code == 409:
        raise GitHubClientError("PR cannot be merged due to conflicts or required checks", status_code=409)
    if response.status_code >= 400:
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        msg = str(data.get("message", f"GitHub merge request failed with status {response.status_code}"))
        raise GitHubClientError(msg, status_code=response.status_code)
    data = response.json()
    return {
        "merged": bool(data.get("merged", False)),
        "message": str(data.get("message", "")),
        "sha": data.get("sha"),
    }


def _parse_branch_ref(payload: dict[str, Any]) -> GitHubBranchRef:
    return GitHubBranchRef(
        ref=str(payload.get("ref", "")),
        sha=str(payload.get("sha", "")),
    )