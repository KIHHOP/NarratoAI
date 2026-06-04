"""
直接上传整段影片给「官方原生大模型」一次性分析的服务。

支持两种官方 provider：
  - gemini: 使用 google-generativeai 的 Files API 上传 → Gemini 模型分析
  - qwen:   使用 DashScope MultiModalConversation 上传本地视频 → Qwen-VL 模型分析

输出格式与抽帧链路保持一致（list[dict]，含 timestamp/picture/narration/OST），
方便 WebUI 与下游剪辑流程透明地复用。
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.config.defaults import (
    DEFAULT_DIRECT_VIDEO_GEMINI_MODEL_NAME,
    DEFAULT_DIRECT_VIDEO_PROVIDER,
    DEFAULT_DIRECT_VIDEO_QWEN_MODEL_NAME,
    DEFAULT_HIGHLIGHT_CLIP_MAX_SECONDS,
    DEFAULT_HIGHLIGHT_CLIP_MIN_SECONDS,
    DEFAULT_HIGHLIGHT_DENSITY_PER_MINUTE,
    DEFAULT_HIGHLIGHT_MODE_ENABLED,
    DIRECT_VIDEO_PROVIDER_GEMINI,
    DIRECT_VIDEO_PROVIDER_QWEN,
)
from app.utils import utils


# ---------- 精华精选模式 Prompt ----------
# 让模型先看完整段视频，再主动从中挑选「精彩瞬间」，
# 而不是均匀地把整段视频切片。这样可以避免一个长镜头被整块吃进成片，
# 让最终成品的信息密度更高。
_HIGHLIGHT_PROMPT = """
你正在协助一档高密度短视频解说节目。请完整观看这段视频后，从中精选最值得保留的「精彩瞬间」用于二次剪辑。

判断「精彩」的标准（按重要性排序）：
1. 信息密度高：动作变化、关键决策、角色情绪转折、画面冲击、转场惊喜。
2. 戏剧张力：冲突、对峙、揭晓、反转、情绪高点。
3. 视觉记忆点：构图独特、特写、慢镜头、爆发性动作。
4. 推动叙事的关键节点：能让观众理解前因后果的最小必要镜头。

必须主动跳过：
- 长时间静止或重复的镜头(即便构图好看)。
- 单纯铺垫、过场、对话拖沓的部分。
- 与主线无关的次要细节。
- 已经出现过的同质内容(同一个动作多次重复时只保留最有代表性的一次)。

请按以下三步流程标注每一个精彩瞬间，**每一步都对应原片真实时间轴(HH:MM:SS,mmm，毫秒补零至 3 位)**：

第一步 highlight_anchor：精彩"爆点"出现的那一帧/那一刻，单一时间点。
  例如某个冲击性动作发生的瞬间、某句关键台词说出口的时刻、某个表情出现的那一帧。

第二步 highlight_window：包含该爆点的完整自然区间。
  例如原片中这个镜头从 00:05:18,000 持续到 00:05:28,000，window 就标这整段。
  即便镜头很长(>10 秒)也照实标注，window 用来表达"这段戏在原片中真实占了多久"。

第三步 timestamp：在 highlight_window 内，挑出最具爆点、信息密度最高的最终切片区间。
  - 最终切片时长建议 ${clip_min_seconds}-${clip_max_seconds} 秒，最长不超过 ${clip_hard_max_seconds} 秒。
  - 必须包含 highlight_anchor。
  - 如果 highlight_window 超过 ${clip_max_seconds} 秒，**只截取 window 中最具爆点的 ${clip_min_seconds}-${clip_max_seconds} 秒**，不要把整段镜头照搬。
  - 这是下游剪辑实际会使用的切片范围。

其它要求：
- 片段之间允许有时间跳跃(不要求连续覆盖整段视频)。
- 全片整体保留密度参考：每分钟原片大约产出 ${density_per_minute} 个精华片段(可按内容松紧浮动 ±50%)。
- 所有时间戳必须严格对应原片真实时间，从 00:00:00,000 起算。

为每个精华片段提供：
- highlight_anchor："HH:MM:SS,mmm" 单点，爆点发生的精确时刻。
- highlight_window："HH:MM:SS,mmm-HH:MM:SS,mmm" 区间，爆点所在的自然镜头/语段在原片占据的完整时长。
- timestamp："HH:MM:SS,mmm-HH:MM:SS,mmm" 区间，在 highlight_window 内挑出的最终切片范围。
- picture：客观陈述该精彩瞬间的画面内容(场景、主体动作、情绪要点)。
- narration：一句节奏紧凑的中文配音稿，约 12-22 字，口语化，承接上下片段。
- highlight_reason：用 6-15 字简述"为什么这一段值得保留"，便于后期剪辑师复核。

${context_block}

输出格式(仅返回 JSON，不要任何额外文字或代码块标记)：
{
  "items": [
    {
      "_id": 1,
      "highlight_anchor": "00:00:01,500",
      "highlight_window": "00:00:00,000-00:00:08,000",
      "timestamp": "00:00:00,000-00:00:03,000",
      "picture": "画面内容",
      "narration": "中文配音稿",
      "highlight_reason": "为什么精彩"
    }
  ]
}
""".strip()



# ---------- 均匀覆盖模式 Prompt（旧行为，作为退路） ----------
_DEFAULT_PROMPT = """
请对这段视频进行内容描述并撰写中文配音稿。

任务步骤：
1. 按时间顺序将视频分为若干连续片段，每段约 ${clip_min_seconds}-${clip_max_seconds} 秒。
2. 为每个片段提供：
   - timestamp：格式 "HH:MM:SS,mmm-HH:MM:SS,mmm"，毫秒补零至 3 位，对应视频真实时间。
   - picture：客观陈述该片段画面中的主要内容（场景、人物、动作）。
   - narration：根据画面写一句中文配音稿，约 15~25 字，自然口语化。
3. 仅基于视频实际内容撰写，不添加视频中未出现的内容。

${context_block}

输出格式（仅返回 JSON，不要任何额外文字或代码块标记）：
{
  "items": [
    {
      "_id": 1,
      "timestamp": "00:00:00,000-00:00:05,000",
      "picture": "画面内容",
      "narration": "中文配音稿"
    }
  ]
}
""".strip()


# 极简 Fallback Prompt：当主 Prompt 触发 block_reason=OTHER 时退化使用
_FALLBACK_PROMPT = """
请观看视频并以 JSON 输出每个精彩片段的描述与对应中文配音稿。

每个片段包含：
- timestamp: "HH:MM:SS,mmm-HH:MM:SS,mmm" 时间区间
- picture: 画面内容描述
- narration: 一句中文配音稿（12-22 字）

每段时长 2-5 秒，仅保留视频中最值得回味的瞬间，可跳过铺垫和重复镜头。仅输出如下 JSON：

{"items":[{"_id":1,"timestamp":"00:00:00,000-00:00:03,000","picture":"...","narration":"..."}]}
""".strip()




# Gemini File API 上传后处理状态轮询
_GEMINI_POLL_INTERVAL_SECONDS = 3
_GEMINI_MAX_PROCESS_WAIT_SECONDS = 600  # 10 分钟

# Gemini generate_content 遇到可恢复错误时的指数退避重试参数
_GEMINI_GENERATE_RETRY_ATTEMPTS = 5
_GEMINI_GENERATE_RETRY_BASE_DELAY = 5  # 秒
# 视为「可重试」的 HTTP 状态码（503 服务超载、429 限流、500 内部错误）
_GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}



class DirectVideoAnalysisService:
    """通过 Gemini / Qwen-VL 官方 API 上传完整视频生成解说脚本。

    采用同步 API 设计：直接在调用方所在线程中阻塞执行，便于 progress_callback
    在 Streamlit 主线程中安全地更新 UI 元素。
    """

    def generate_script(
        self,
        *,
        video_path: str,
        video_theme: str = "",
        custom_prompt: str = "",
        provider: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        highlight_mode_enabled: bool | None = None,
        highlight_clip_min_seconds: float | None = None,
        highlight_clip_max_seconds: float | None = None,
        highlight_density_per_minute: float | None = None,
    ) -> list[dict[str, Any]]:
        progress = progress_callback or (lambda _p, _m: None)

        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        provider = (provider or config.app.get("direct_video_provider") or DEFAULT_DIRECT_VIDEO_PROVIDER).lower()
        if provider not in (DIRECT_VIDEO_PROVIDER_GEMINI, DIRECT_VIDEO_PROVIDER_QWEN):
            raise ValueError(
                f"未知的 direct_video_provider: {provider}，仅支持: {DIRECT_VIDEO_PROVIDER_GEMINI} / {DIRECT_VIDEO_PROVIDER_QWEN}"
            )

        api_key, model_name = self._resolve_credentials(provider, api_key, model_name)

        highlight_settings = self._resolve_highlight_settings(
            highlight_mode_enabled=highlight_mode_enabled,
            highlight_clip_min_seconds=highlight_clip_min_seconds,
            highlight_clip_max_seconds=highlight_clip_max_seconds,
            highlight_density_per_minute=highlight_density_per_minute,
        )
        logger.info(
            "Direct Video 生成参数 | highlight_mode={enabled} clip={min:.1f}-{max:.1f}s density={density:.1f}/min".format(
                enabled=highlight_settings["enabled"],
                min=highlight_settings["clip_min_seconds"],
                max=highlight_settings["clip_max_seconds"],
                density=highlight_settings["density_per_minute"],
            )
        )

        prompt = self._build_prompt(
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            highlight_settings=highlight_settings,
        )


        if provider == DIRECT_VIDEO_PROVIDER_GEMINI:
            raw_text = self._run_gemini(
                video_path=video_path,
                prompt=prompt,
                api_key=api_key,
                model_name=model_name,
                progress=progress,
            )
        else:
            raw_text = self._run_qwen(
                video_path=video_path,
                prompt=prompt,
                api_key=api_key,
                model_name=model_name,
                progress=progress,
            )

        progress(85, "正在解析解说脚本...")
        script_items = self._parse_response_items(
            raw_text,
            highlight_settings=highlight_settings,
        )

        # 保存原始产物，便于排查
        self._save_artifact(
            video_path=video_path,
            provider=provider,
            model_name=model_name,
            raw_text=raw_text,
            script_items=script_items,
            highlight_settings=highlight_settings,
        )

        final_script = [{**item, "OST": 2} for item in script_items]
        progress(100, "脚本生成完成")
        return final_script



    # ---------- Provider 配置解析 ----------
    def _resolve_credentials(
        self,
        provider: str,
        api_key: str | None,
        model_name: str | None,
    ) -> tuple[str, str]:
        if provider == DIRECT_VIDEO_PROVIDER_GEMINI:
            api_key = api_key or config.app.get("direct_video_gemini_api_key", "")
            model_name = model_name or config.app.get("direct_video_gemini_model_name") or DEFAULT_DIRECT_VIDEO_GEMINI_MODEL_NAME
            human = "Gemini"
        else:
            api_key = api_key or config.app.get("direct_video_qwen_api_key", "")
            model_name = model_name or config.app.get("direct_video_qwen_model_name") or DEFAULT_DIRECT_VIDEO_QWEN_MODEL_NAME
            human = "Qwen-VL"

        if not api_key:
            raise ValueError(
                f"未配置 {human} 的 API Key，请在「基础设置 → 视频分析模式 → 直接上传分析」中填写"
            )
        if not model_name:
            raise ValueError(f"未配置 {human} 的模型名称")
        return api_key, model_name

    # ---------- Gemini 官方（google-genai 新版 SDK）----------
    def _run_gemini(
        self,
        *,
        video_path: str,
        prompt: str,
        api_key: str,
        model_name: str,
        progress: Callable[[float, str], None],
    ) -> str:
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "未安装 google-genai，无法使用 Gemini 直接视频分析。"
                "请执行：pip install google-genai"
            ) from exc

        client = google_genai.Client(api_key=api_key)

        progress(15, "正在上传视频到 Gemini File API...")
        upload_name = os.path.basename(video_path)
        try:
            video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"待上传视频: {video_path} (大小 {video_size_mb:.2f} MB)")
        except OSError:
            pass
        uploaded = client.files.upload(
            file=video_path,
            config=genai_types.UploadFileConfig(display_name=upload_name),
        )
        uploaded_name = getattr(uploaded, "name", None) or ""
        initial_state = getattr(getattr(uploaded, "state", None), "name", "") or str(
            getattr(uploaded, "state", "")
        )
        logger.info(f"已上传视频到 Gemini: {uploaded_name} ({upload_name}), 初始状态={initial_state or 'UNKNOWN'}")

        progress(35, "等待 Gemini 完成视频预处理（PROCESSING → ACTIVE）...")
        active_file = self._wait_until_active_gemini(client, uploaded_name)
        logger.info(f"Gemini 视频预处理完成，文件状态已变为 ACTIVE: {uploaded_name}")


        progress(60, "Gemini 正在分析视频内容...")
        # 把所有安全分类设为 BLOCK_NONE，避免误判（特别是 block_reason=OTHER）
        safety_settings = self._build_gemini_safety_settings(genai_types)
        gen_config = genai_types.GenerateContentConfig(
            temperature=0.8,
            response_mime_type="application/json",
            # 长视频会产生很多片段，提高输出上限避免被截断
            max_output_tokens=32768,
            safety_settings=safety_settings,
        )

        # 主 prompt 调用
        try:
            response = self._gemini_generate_with_retry(
                client=client,
                model_name=model_name,
                contents=[active_file, prompt],
                generation_config=gen_config,
                progress=progress,
            )
            raw_text = self._extract_gemini_response_text(response)
        except RuntimeError as exc:
            # 主 prompt 触发了 prompt-level 阻挡（block_reason=OTHER 等不可控分类），
            # 自动换成极简中性 prompt 再试一次。
            if not self._is_prompt_block_error(exc):
                raise
            logger.warning(f"主 prompt 被 Gemini 阻挡，自动改用 fallback prompt 重试: {exc}")
            progress(70, "提示词被安全策略阻挡，正在使用极简提示词重试...")
            response = self._gemini_generate_with_retry(
                client=client,
                model_name=model_name,
                contents=[active_file, _FALLBACK_PROMPT],
                generation_config=gen_config,
                progress=progress,
            )
            raw_text = self._extract_gemini_response_text(response)






        # 清理远端文件，避免占用配额（失败不影响主流程）
        try:
            if uploaded_name:
                client.files.delete(name=uploaded_name)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"清理 Gemini 远端文件失败: {exc}")

        return raw_text

    def _wait_until_active_gemini(self, client: Any, file_name: str):
        if not file_name:
            raise RuntimeError("Gemini 上传后未返回 file name，无法继续")

        elapsed = 0
        while elapsed < _GEMINI_MAX_PROCESS_WAIT_SECONDS:
            file_info = client.files.get(name=file_name)
            state = getattr(getattr(file_info, "state", None), "name", "") or str(
                getattr(file_info, "state", "")
            )
            if state.upper() == "ACTIVE":
                return file_info
            if state.upper() == "FAILED":
                raise RuntimeError(f"Gemini 处理视频失败: {file_name}")
            time.sleep(_GEMINI_POLL_INTERVAL_SECONDS)
            elapsed += _GEMINI_POLL_INTERVAL_SECONDS

        raise TimeoutError(
            f"等待 Gemini 处理视频超时（>{_GEMINI_MAX_PROCESS_WAIT_SECONDS}s）: {file_name}"
        )

    def _extract_gemini_response_text(self, response: Any) -> str:
        # 1) 优先用 SDK 提供的 .text 便利属性
        text = getattr(response, "text", None)
        if text:
            return text

        # 2) 退而手动遍历 candidates，把所有 part.text 拼起来（即使被截断也尽量保留）
        candidates = getattr(response, "candidates", None) or []
        finish_reason: str = ""
        collected: list[str] = []
        safety_blocked = False

        for candidate in candidates:
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reason = getattr(fr, "name", None) or str(fr)

            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", "") or ""
                if part_text:
                    collected.append(part_text)

            # 检查 safety ratings 是否触发 BLOCK
            safety_ratings = getattr(candidate, "safety_ratings", None) or []
            for rating in safety_ratings:
                if getattr(rating, "blocked", False):
                    safety_blocked = True

        # 3) 检查 prompt_feedback（可能整段被 prompt 安全策略阻挡）
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = ""
        if prompt_feedback is not None:
            br = getattr(prompt_feedback, "block_reason", None)
            if br is not None:
                block_reason = getattr(br, "name", None) or str(br)

        if collected:
            joined = "".join(collected)
            if finish_reason and finish_reason.upper() == "MAX_TOKENS":
                logger.warning(
                    "Gemini 输出被 MAX_TOKENS 截断，已保留部分内容尝试解析。"
                    "建议：缩短视频时长、或在模型支持时增加 max_output_tokens。"
                )
            return joined

        # 4) 没有任何 part.text，根据原因抛精准错误
        reason_upper = (finish_reason or "").upper()
        if reason_upper == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini 输出被 MAX_TOKENS 截断且没有任何可解析片段。"
                "请尝试：1) 缩短视频时长 2) 改用更大输出窗口的模型（如 gemini-1.5-pro / gemini-2.5-pro）"
            )
        if reason_upper in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII"} or safety_blocked:
            raise RuntimeError(
                f"Gemini 因安全/合规策略未返回内容（finish_reason={finish_reason or 'SAFETY'}）。"
                "请检查视频内容是否触发安全策略，或调整 prompt。"
            )
        if block_reason:
            raise RuntimeError(
                f"Gemini 提示词被安全策略阻挡（block_reason={block_reason}），未生成任何输出。"
            )
        if reason_upper:
            raise RuntimeError(
                f"Gemini 未返回可解析的文本内容（finish_reason={finish_reason}）。"
            )
        raise RuntimeError("Gemini 未返回可解析的文本内容")


    def _gemini_generate_with_retry(
        self,
        *,
        client: Any,
        model_name: str,
        contents: list,
        generation_config: Any,
        progress: Callable[[float, str], None],
    ) -> Any:
        """对 Gemini generate_content 增加指数退避重试，避免被 503/429 类瞬时错误打断。"""
        last_exc: Exception | None = None
        for attempt in range(1, _GEMINI_GENERATE_RETRY_ATTEMPTS + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

            except Exception as exc:
                last_exc = exc
                status_code = self._extract_gemini_status_code(exc)
                # 只对可恢复的错误（503/429/500/502/504）重试
                if status_code not in _GEMINI_RETRYABLE_STATUS_CODES:
                    raise
                if attempt >= _GEMINI_GENERATE_RETRY_ATTEMPTS:
                    break

                delay = _GEMINI_GENERATE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Gemini generate_content 临时不可用 (status={status_code}, attempt={attempt}/"
                    f"{_GEMINI_GENERATE_RETRY_ATTEMPTS})，{delay}s 后重试: {exc}"
                )
                progress(
                    65,
                    f"Gemini 暂时繁忙（HTTP {status_code}），{delay}s 后自动重试 "
                    f"({attempt}/{_GEMINI_GENERATE_RETRY_ATTEMPTS - 1})...",
                )
                time.sleep(delay)

        # 全部重试都失败
        if last_exc is None:
            raise RuntimeError("Gemini generate_content 重试失败但未捕获到原始异常")
        status_code = self._extract_gemini_status_code(last_exc)
        if status_code == 503:
            raise RuntimeError(
                "Gemini 服务当前需求过高 (503 UNAVAILABLE)，多次重试仍失败。"
                "建议稍后再试，或切换到 Qwen-VL 提供商。"
            ) from last_exc
        if status_code == 429:
            raise RuntimeError(
                "Gemini 调用超出速率限制 (429)，多次重试仍失败。"
                "请稍后再试或检查 API Key 的配额。"
            ) from last_exc
        raise last_exc

    @staticmethod
    def _extract_gemini_status_code(exc: Exception) -> int | None:
        """从 google-genai 抛出的异常中提取 HTTP status code。"""
        # 新版 SDK 的 APIError 直接带 status_code 属性
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        # 字符串里通常会带 "503 UNAVAILABLE"，再做一次保底解析
        msg = str(exc)
        match = re.match(r"\s*(\d{3})\s+", msg)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _is_prompt_block_error(exc: Exception) -> bool:
        """判断异常是否为 prompt-level 阻挡（block_reason=OTHER 等不可控分类）。

        这种错误重试同一个 prompt 不会成功，需要换 fallback prompt。
        """
        msg = str(exc) or ""
        keywords = [
            "block_reason=OTHER",
            "block_reason=BLOCKLIST",
            "block_reason=PROHIBITED",
            "block_reason=SPII",
            "提示词被安全策略阻挡",
            "未生成任何输出",
        ]
        return any(keyword in msg for keyword in keywords)

    @staticmethod
    def _build_gemini_safety_settings(genai_types: Any) -> list[Any]:

        """构造把所有安全分类都设为 BLOCK_NONE 的 safety_settings。

        Gemini 默认的安全过滤偶尔会以 block_reason=OTHER 把整个 prompt 拒绝；
        视频解说脚本生成是创作类用途，把过滤开到最宽即可。
        如果新版 SDK 类型枚举有变动，捕获异常退化为不传 safety_settings。
        """
        try:
            HarmCategory = genai_types.HarmCategory
            HarmBlockThreshold = genai_types.HarmBlockThreshold
            categories = [
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
            # CIVIC_INTEGRITY 在较新版本才有，单独 try
            civic = getattr(HarmCategory, "HARM_CATEGORY_CIVIC_INTEGRITY", None)
            if civic is not None:
                categories.append(civic)
            return [
                genai_types.SafetySetting(
                    category=category,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                )
                for category in categories
            ]
        except Exception as exc:  # pragma: no cover - SDK 兼容性
            logger.warning(f"无法构造 Gemini safety_settings，将使用默认安全策略: {exc}")
            return []


    # ---------- Qwen-VL（DashScope）----------


    def _run_qwen(
        self,
        *,
        video_path: str,
        prompt: str,
        api_key: str,
        model_name: str,
        progress: Callable[[float, str], None],
    ) -> str:
        try:
            from dashscope import MultiModalConversation
        except ImportError as exc:
            raise RuntimeError(
                "未安装 dashscope，无法使用 Qwen-VL 直接视频分析。"
                "请在 requirements.txt 中保留 dashscope 依赖"
            ) from exc

        progress(20, "正在准备视频文件...")
        # DashScope 支持本地文件 URI: file://<absolute_path>
        absolute_path = os.path.abspath(video_path)
        # 兼容 Windows 路径
        absolute_path_normalized = absolute_path.replace("\\", "/")
        if not absolute_path_normalized.startswith("/"):
            # 例如 D:/video.mp4 → /D:/video.mp4
            absolute_path_normalized = "/" + absolute_path_normalized
        video_uri = f"file://{absolute_path_normalized}"
        logger.info(f"Qwen-VL 视频 URI: {video_uri}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_uri},
                    {"text": prompt},
                ],
            }
        ]

        progress(45, "Qwen-VL 正在分析视频内容（可能需要数分钟）...")
        response = MultiModalConversation.call(
            api_key=api_key,
            model=model_name,
            messages=messages,
        )
        return self._extract_qwen_response_text(response)

    def _extract_qwen_response_text(self, response: Any) -> str:
        # 失败状态
        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code != 200:
            err_code = getattr(response, "code", "")
            err_msg = getattr(response, "message", "")
            raise RuntimeError(f"Qwen-VL 调用失败 (status={status_code}, code={err_code}): {err_msg}")

        output = getattr(response, "output", None)
        if not output:
            raise RuntimeError("Qwen-VL 未返回 output")

        # output.choices[0].message.content 可能是字符串或 list[dict]
        choices = output.get("choices") if isinstance(output, dict) else getattr(output, "choices", None)
        if not choices:
            raise RuntimeError("Qwen-VL 返回 output 中没有 choices")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if not message:
            raise RuntimeError("Qwen-VL 返回的 choice 没有 message")

        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text") or ""
                    if part_text:
                        collected.append(str(part_text))
                else:
                    collected.append(str(part))
            text = "".join(collected)
            if text.strip():
                return text
        raise RuntimeError("Qwen-VL 返回内容为空或无法解析")

    # ---------- 精华模式参数 ----------
    @staticmethod
    def _resolve_highlight_settings(
        *,
        highlight_mode_enabled: bool | None,
        highlight_clip_min_seconds: float | None,
        highlight_clip_max_seconds: float | None,
        highlight_density_per_minute: float | None,
    ) -> dict[str, Any]:
        """合并显式参数 / 配置 / 默认值，返回标准化后的精华模式参数。"""

        def _coerce_bool(value: Any, fallback: bool) -> bool:
            if value is None:
                return fallback
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return fallback

        def _coerce_float(value: Any, fallback: float) -> float:
            if value is None:
                return float(fallback)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(fallback)

        enabled_default = bool(
            config.app.get("highlight_mode_enabled", DEFAULT_HIGHLIGHT_MODE_ENABLED)
        )
        clip_min_default = _coerce_float(
            config.app.get("highlight_clip_min_seconds"), DEFAULT_HIGHLIGHT_CLIP_MIN_SECONDS
        )
        clip_max_default = _coerce_float(
            config.app.get("highlight_clip_max_seconds"), DEFAULT_HIGHLIGHT_CLIP_MAX_SECONDS
        )
        density_default = _coerce_float(
            config.app.get("highlight_density_per_minute"), DEFAULT_HIGHLIGHT_DENSITY_PER_MINUTE
        )

        enabled = _coerce_bool(highlight_mode_enabled, enabled_default)
        clip_min = _coerce_float(highlight_clip_min_seconds, clip_min_default)
        clip_max = _coerce_float(highlight_clip_max_seconds, clip_max_default)
        density = _coerce_float(highlight_density_per_minute, density_default)

        # 健全性约束：min < max，且都在合理范围内
        clip_min = max(0.5, min(clip_min, 60.0))
        clip_max = max(0.5, min(clip_max, 60.0))
        if clip_max < clip_min:
            clip_max = clip_min
        # 硬上限给 prompt 用，比建议上限再宽 50%（但不超过 30 秒）
        clip_hard_max = min(clip_max * 1.5, 30.0)
        density = max(0.5, min(density, 30.0))

        return {
            "enabled": enabled,
            "clip_min_seconds": clip_min,
            "clip_max_seconds": clip_max,
            "clip_hard_max_seconds": clip_hard_max,
            "density_per_minute": density,
        }

    # ---------- 通用解析 / 落盘 ----------
    def _build_prompt(
        self,
        *,
        video_theme: str,
        custom_prompt: str,
        highlight_settings: dict[str, Any] | None = None,
    ) -> str:
        context_lines: list[str] = []
        if (video_theme or "").strip():
            context_lines.append(f"视频主题：{video_theme.strip()}")
        if (custom_prompt or "").strip():
            context_lines.append(f"补充创作要求：{custom_prompt.strip()}")

        context_block = ""
        if context_lines:
            joined = "\n".join(f"- {line}" for line in context_lines)
            context_block = f"创作上下文：\n{joined}"

        settings = highlight_settings or self._resolve_highlight_settings(
            highlight_mode_enabled=None,
            highlight_clip_min_seconds=None,
            highlight_clip_max_seconds=None,
            highlight_density_per_minute=None,
        )

        template = _HIGHLIGHT_PROMPT if settings["enabled"] else _DEFAULT_PROMPT

        clip_min = settings["clip_min_seconds"]
        clip_max = settings["clip_max_seconds"]
        clip_hard_max = settings["clip_hard_max_seconds"]
        density = settings["density_per_minute"]

        return (
            template
            .replace("${context_block}", context_block)
            .replace("${clip_min_seconds}", f"{clip_min:.1f}")
            .replace("${clip_max_seconds}", f"{clip_max:.1f}")
            .replace("${clip_hard_max_seconds}", f"{clip_hard_max:.1f}")
            .replace("${density_per_minute}", f"{density:.1f}")
        )


    # ---------- 时间戳工具 ----------
    @staticmethod
    def _ts_to_ms(value: str) -> int | None:
        """把 'HH:MM:SS,mmm' 或 'HH:MM:SS.mmm' / 'HH:MM:SS' 解析为毫秒。"""
        if not value:
            return None
        text = value.strip().replace(".", ",")
        m = re.match(r"^(\d{1,2}):(\d{1,2}):(\d{1,2})(?:,(\d{1,3}))?$", text)
        if not m:
            return None
        h, mm, s, ms = m.group(1), m.group(2), m.group(3), m.group(4) or "0"
        try:
            ms_padded = (ms + "000")[:3]
            return int(h) * 3600_000 + int(mm) * 60_000 + int(s) * 1000 + int(ms_padded)
        except ValueError:
            return None

    @staticmethod
    def _ms_to_ts(ms: int) -> str:
        ms = max(0, int(ms))
        h, rem = divmod(ms, 3600_000)
        mm, rem = divmod(rem, 60_000)
        s, ms_part = divmod(rem, 1000)
        return f"{h:02d}:{mm:02d}:{s:02d},{ms_part:03d}"

    @classmethod
    def _parse_range(cls, value: str) -> tuple[int, int] | None:
        """解析 'start-end' 区间为 (start_ms, end_ms)。"""
        if not value or "-" not in value:
            return None
        parts = value.split("-", 1)
        if len(parts) != 2:
            return None
        start_ms = cls._ts_to_ms(parts[0])
        end_ms = cls._ts_to_ms(parts[1])
        if start_ms is None or end_ms is None:
            return None
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        return start_ms, end_ms

    @classmethod
    def _refine_clip_range(
        cls,
        *,
        timestamp_range: tuple[int, int] | None,
        anchor_ms: int | None,
        window_range: tuple[int, int] | None,
        clip_min_seconds: float,
        clip_max_seconds: float,
        clip_hard_max_seconds: float,
    ) -> tuple[tuple[int, int], dict[str, Any]] | None:
        """根据 anchor / window 校正最终切片区间。

        优先使用模型给的 timestamp，若违反约束则按 anchor 中心 + clip_min/max 重新生成。
        返回 (refined_range_ms, info_dict)；info 用于日志与 artifact 落盘。
        """
        clip_min_ms = int(round(clip_min_seconds * 1000))
        clip_max_ms = int(round(clip_max_seconds * 1000))
        clip_hard_max_ms = int(round(clip_hard_max_seconds * 1000))

        adjustments: list[str] = []

        ts_range = timestamp_range
        if ts_range is None and window_range is None and anchor_ms is None:
            return None

        # 先决定一个候选区间
        candidate: tuple[int, int]
        if ts_range is not None:
            candidate = ts_range
        elif window_range is not None:
            candidate = window_range
        else:
            # 仅有 anchor：以 anchor 为中心生成 clip_max 长度区间
            half = clip_max_ms // 2
            candidate = (max(0, anchor_ms - half), anchor_ms + (clip_max_ms - half))
            adjustments.append("derived_from_anchor")

        # 限制候选区间不超过硬上限
        c_start, c_end = candidate
        if c_end - c_start > clip_hard_max_ms:
            adjustments.append("clamp_to_hard_max")
            # 优先以 anchor 为中心收敛；没 anchor 就从头收敛
            if anchor_ms is not None and c_start <= anchor_ms <= c_end:
                half = clip_max_ms // 2
                c_start_new = max(c_start, anchor_ms - half)
                c_end_new = min(c_end, c_start_new + clip_max_ms)
                c_start, c_end = c_start_new, c_end_new
            else:
                c_end = c_start + clip_max_ms

        # 确保 anchor 落在候选区间内
        if anchor_ms is not None and not (c_start <= anchor_ms <= c_end):
            adjustments.append("anchor_outside_clip")
            half = clip_max_ms // 2
            c_start = max(0, anchor_ms - half)
            c_end = c_start + clip_max_ms
            # 如果 window 存在，确保候选不超出 window
            if window_range is not None:
                w_start, w_end = window_range
                if c_start < w_start:
                    c_start = w_start
                    c_end = min(w_end, c_start + clip_max_ms)
                if c_end > w_end:
                    c_end = w_end
                    c_start = max(w_start, c_end - clip_max_ms)

        # 区间过短：扩展到 clip_min
        if c_end - c_start < clip_min_ms:
            adjustments.append("extend_to_min")
            need = clip_min_ms - (c_end - c_start)
            # 优先把缺口补在右侧；不行再补左侧
            c_end_new = c_end + need
            if window_range is not None and c_end_new > window_range[1]:
                overflow = c_end_new - window_range[1]
                c_end_new = window_range[1]
                c_start = max(0, c_start - overflow)
            c_end = c_end_new

        # 限制不超过 clip_max（软）：剪掉超出 anchor 中心的部分
        if c_end - c_start > clip_max_ms:
            adjustments.append("clamp_to_soft_max")
            if anchor_ms is not None and c_start <= anchor_ms <= c_end:
                half = clip_max_ms // 2
                c_start = max(c_start, anchor_ms - half)
                c_end = c_start + clip_max_ms
            else:
                c_end = c_start + clip_max_ms

        c_start = max(0, c_start)
        c_end = max(c_start + 1, c_end)

        info = {
            "adjustments": adjustments,
            "duration_ms": c_end - c_start,
        }
        return (c_start, c_end), info

    # ---------- 解析 ----------
    def _parse_response_items(
        self,
        raw_text: str,
        *,
        highlight_settings: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cleaned = (raw_text or "").strip()
        if not cleaned:
            raise ValueError("模型返回内容为空")

        # 去除 ``` 代码块包裹
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = self._loads_truncated_json(cleaned)
            if payload is None:
                raise ValueError(f"无法解析模型返回的 JSON: {cleaned[:200]}")

        if not isinstance(payload, dict):
            raise ValueError("模型返回 JSON 必须是对象")

        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("模型返回的 JSON 缺少非空的 items 数组")

        # 没传 highlight_settings（例如外部直接调用 _parse_response_items 测试）也能跑
        if highlight_settings is None:
            highlight_settings = self._resolve_highlight_settings(
                highlight_mode_enabled=None,
                highlight_clip_min_seconds=None,
                highlight_clip_max_seconds=None,
                highlight_density_per_minute=None,
            )

        clip_min_seconds = float(highlight_settings.get("clip_min_seconds", DEFAULT_HIGHLIGHT_CLIP_MIN_SECONDS))
        clip_max_seconds = float(highlight_settings.get("clip_max_seconds", DEFAULT_HIGHLIGHT_CLIP_MAX_SECONDS))
        clip_hard_max_seconds = float(highlight_settings.get("clip_hard_max_seconds", clip_max_seconds * 1.5))

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            raw_timestamp = str(item.get("timestamp", "")).strip()
            raw_anchor = str(item.get("highlight_anchor", "")).strip()
            raw_window = str(item.get("highlight_window", "")).strip()
            picture = str(item.get("picture", "")).strip()
            narration = str(item.get("narration", "")).strip()
            highlight_reason = str(item.get("highlight_reason", "")).strip()

            timestamp_range = self._parse_range(raw_timestamp) if raw_timestamp else None
            anchor_ms = self._ts_to_ms(raw_anchor) if raw_anchor else None
            window_range = self._parse_range(raw_window) if raw_window else None

            # 若三者都拿不到，等同于片段缺失，跳过
            if timestamp_range is None and window_range is None and anchor_ms is None:
                logger.warning(f"第 {index} 个片段缺少有效时间信息，已跳过")
                continue

            refined = self._refine_clip_range(
                timestamp_range=timestamp_range,
                anchor_ms=anchor_ms,
                window_range=window_range,
                clip_min_seconds=clip_min_seconds,
                clip_max_seconds=clip_max_seconds,
                clip_hard_max_seconds=clip_hard_max_seconds,
            )
            if refined is None:
                logger.warning(f"第 {index} 个片段无法计算切片范围，已跳过")
                continue
            (refined_start_ms, refined_end_ms), refine_info = refined
            refined_timestamp = (
                f"{self._ms_to_ts(refined_start_ms)}-{self._ms_to_ts(refined_end_ms)}"
            )

            entry: dict[str, Any] = {
                "_id": item.get("_id", index),
                "timestamp": refined_timestamp,
                "picture": picture,
                "narration": narration,
            }
            # 保留模型给的原始时间信息，便于 UI 展示与重新切片
            if anchor_ms is not None:
                entry["highlight_anchor"] = self._ms_to_ts(anchor_ms)
            if window_range is not None:
                entry["highlight_window"] = (
                    f"{self._ms_to_ts(window_range[0])}-{self._ms_to_ts(window_range[1])}"
                )
            if highlight_reason:
                entry["highlight_reason"] = highlight_reason
            # 如果切片被微调过，保留模型原始 timestamp 作为参考
            if timestamp_range is not None and (refined_start_ms, refined_end_ms) != timestamp_range:
                entry["timestamp_raw"] = raw_timestamp
            if refine_info["adjustments"]:
                entry["timestamp_adjustments"] = refine_info["adjustments"]

            normalized.append(entry)

        if not normalized:
            raise ValueError("模型返回的 items 中没有有效片段")

        adjusted = sum(1 for it in normalized if it.get("timestamp_adjustments"))
        if adjusted:
            logger.info(
                f"精华模式 timestamp 校正：{adjusted}/{len(normalized)} 个片段被调整到配置区间内"
            )
        return normalized



    @staticmethod
    def _loads_truncated_json(text: str) -> dict | None:
        """容错解析被 MAX_TOKENS 截断的 JSON：

        策略：
        1. 取从第一个 `{` 开始的子串
        2. 若直接 `json.loads` 失败，逐步剪掉末尾不完整的字符并补齐 `]` `}`
        3. 仍失败则返回 None
        """
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        body = text[start:]

        # 直接尝试
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

        # 截断到最后一个完整的 `}`（试图回退到一个完整 item）
        for end in range(len(body) - 1, 0, -1):
            ch = body[end]
            if ch != "}":
                continue
            head = body[: end + 1]
            # 计算未闭合的中括号 / 大括号差额，进行补齐
            opens_brace = head.count("{")
            closes_brace = head.count("}")
            opens_bracket = head.count("[")
            closes_bracket = head.count("]")
            patched = head
            patched += "]" * max(0, opens_bracket - closes_bracket)
            patched += "}" * max(0, opens_brace - closes_brace)
            # 去掉常见的尾随逗号
            patched = re.sub(r",\s*([\]}])", r"\1", patched)
            try:
                parsed = json.loads(patched)
                logger.warning(
                    f"模型 JSON 被截断，已通过补齐括号恢复（保留到偏移 {end + 1} / {len(body)}）"
                )
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue

        return None

    def _save_artifact(
        self,
        *,
        video_path: str,
        provider: str,
        model_name: str,
        raw_text: str,
        script_items: list[dict[str, Any]],
        highlight_settings: dict[str, Any] | None = None,
    ) -> str:
        analysis_dir = os.path.join(utils.storage_dir(), "temp", "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        filename = f"direct_video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(analysis_dir, filename)
        artifact = {
            "artifact_version": "documentary-direct-video-v5",

            "generated_at": datetime.now().isoformat(),
            "video_path": video_path,
            "provider": provider,
            "model_name": model_name,
            "highlight_settings": highlight_settings or {},
            "raw_response": raw_text,
            "script_items": script_items,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(artifact, fp, ensure_ascii=False, indent=2)
            logger.info(f"直接视频分析结果已保存到: {file_path}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"保存直接视频分析结果失败: {exc}")
        return file_path
