"use client";

import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bookmark,
  BookmarkCheck,
  Check,
  ChevronDown,
  ChevronUp,
  CornerDownRight,
  ExternalLink,
  FileText,
  Loader2,
} from "lucide-react";
import {
  CHARACTERS,
  DOMAIN_LABELS,
  EVIDENCE_STYLE,
  FALLBACK_OFFICIAL_SITES,
  getCharacterForDomain,
} from "@/lib/domains";
import type { ChatTurn, EvidenceState, UserHousingProfile } from "@/lib/types";

interface ChatBubbleProps {
  turn: ChatTurn;
  isLatest: boolean;
  conversationalMode: boolean;
  userProfile: UserHousingProfile;
  isBookmarked: boolean;
  onToggleBookmark: (turn: ChatTurn) => void;
  onFollowUpClick: (question: string) => void;
}

interface InvestigationStage {
  label: string;
  caption: string;
  visual: string;
  isScene: boolean;
}

const INVESTIGATION_STAGES: InvestigationStage[] = [
  {
    label: "질문 의도 및 직전 대화 맥락 확인",
    caption: "질문 핵심과 앞선 대화 맥락을 정리하고 있어.",
    visual: "/characters/conan_thinking.png",
    isScene: false,
  },
  {
    label: "공고문·표 원문에서 단서 탐색 중",
    caption: "돋보기로 공고문과 표 단서를 샅샅이 찾는 중!",
    visual: "/scenes/conan_searching.jpg",
    isScene: true,
  },
  {
    label: "핵심 근거 선별 및 숫자·조건 대조",
    caption: "찾아낸 상위 근거의 금액과 기준을 꼼꼼히 대조하고 있어.",
    visual: "/scenes/conan_searching.jpg",
    isScene: true,
  },
  {
    label: "답변 정리 및 마무리",
    caption: "단서 확인 완료! 보기 좋게 정리해서 바로 알려줄게.",
    visual: "/characters/conan_happy.png",
    isScene: false,
  },
];

function formatInlineCitations(text: string): string {
  const normalized = text.replace(/\*\*\[(\d+)\]\*\*/g, "`[$1]`");
  return normalized.replace(/(?<!`)(?<!\])\[(\d+)\](?!\()(?!`)/g, "`[$1]`");
}

function useTypewriter(fullText: string, enabled: boolean) {
  const [shown, setShown] = useState(enabled ? "" : fullText);

  useEffect(() => {
    if (!enabled) {
      setShown(fullText);
      return;
    }
    setShown("");
    if (!fullText) return;

    const tokens = fullText.match(/\S+\s*|\n+/g) ?? [fullText];
    let idx = 0;
    const step = tokens.length > 90 ? 3 : 2;
    const timer = window.setInterval(() => {
      idx = Math.min(idx + step, tokens.length);
      setShown(tokens.slice(0, idx).join(""));
      if (idx >= tokens.length) {
        window.clearInterval(timer);
      }
    }, 16);

    return () => window.clearInterval(timer);
  }, [fullText, enabled]);

  return shown;
}

function InvestigationStepper({ characterName }: { characterName: string }) {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timers = [
      window.setTimeout(() => setActiveStep(1), 1400),
      window.setTimeout(() => setActiveStep(2), 4200),
      window.setTimeout(() => setActiveStep(3), 7200),
    ];
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, []);

  const stage = INVESTIGATION_STAGES[activeStep];

  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
      {/* Situational Anime / 3D Cut-in Visual */}
      <div className="relative h-32 w-full shrink-0 overflow-hidden rounded-xl border border-[#E7E7E2] bg-[#F8F8F6] sm:h-32 sm:w-44">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${activeStep}-${stage.visual}`}
            initial={{ opacity: 0, scale: 1.04 }}
            animate={{ opacity: 1, scale: stage.isScene ? 1.07 : 1 }}
            exit={{ opacity: 0 }}
            transition={{
              opacity: { duration: 0.25 },
              scale: { duration: 2.6, ease: "easeOut" },
            }}
            className="flex h-full w-full items-center justify-center"
          >
            <img
              src={stage.visual}
              alt={stage.label}
              className={
                stage.isScene
                  ? "h-full w-full object-cover object-center"
                  : "h-full w-full object-contain p-2"
              }
            />
          </motion.div>
        </AnimatePresence>
        <div className="absolute bottom-1.5 left-1.5 rounded-md bg-black/65 px-2 py-0.5 text-[10px] font-bold text-white backdrop-blur-sm">
          STEP {activeStep + 1} / {INVESTIGATION_STAGES.length}
        </div>
      </div>

      {/* Live Step Checklist + Spoken Caption */}
      <div className="min-w-0 flex-1 space-y-2.5">
        <div>
          <div className="text-[13.5px] font-bold text-zinc-900">
            {characterName}: &ldquo;{stage.caption}&rdquo;
          </div>
        </div>
        <div className="space-y-1.5">
          {INVESTIGATION_STAGES.map((item, idx) => {
            const done = idx < activeStep;
            const current = idx === activeStep;
            return (
              <div
                key={item.label}
                className={`flex items-center gap-2 text-xs transition-colors ${
                  done
                    ? "text-zinc-500"
                    : current
                      ? "font-bold text-zinc-900"
                      : "text-zinc-300"
                }`}
              >
                {done ? (
                  <Check className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
                ) : current ? (
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-800" />
                ) : (
                  <span className="ml-1 mr-1 h-1.5 w-1.5 shrink-0 rounded-full bg-zinc-200" />
                )}
                <span>{item.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function ChatBubble({
  turn,
  isLatest,
  conversationalMode,
  userProfile,
  isBookmarked,
  onToggleBookmark,
  onFollowUpClick,
}: ChatBubbleProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  const resp = turn.response;
  const domain = resp?.domain || turn.selectedPart;
  const character = useMemo(() => {
    if (resp?.character_id && CHARACTERS[resp.character_id]) {
      return CHARACTERS[resp.character_id];
    }
    return getCharacterForDomain(domain, turn.selectedPart);
  }, [resp?.character_id, domain, turn.selectedPart]);

  const evidenceState: EvidenceState = useMemo(() => {
    if (!resp) return "grounded";
    if (resp.meta?.evidence_state) return resp.meta.evidence_state;
    if (resp.status === "error") return "error";
    if (resp.status === "answered") return "grounded";
    if (resp.status === "unsupported") return "no_evidence";
    return "partial_missing";
  }, [resp]);

  const portraitSrc = useMemo(() => {
    if (evidenceState === "partial_missing" || evidenceState === "no_evidence") {
      return character.thinkingAvatar || character.avatar;
    }
    return character.avatar;
  }, [character, evidenceState]);

  const formattedAnswer = useMemo(
    () => formatInlineCitations(resp?.answer || ""),
    [resp?.answer]
  );

  const displayedAnswer = useTypewriter(
    formattedAnswer,
    Boolean(turn.isNew && conversationalMode && resp && !turn.isLoading)
  );

  const statusBadge = EVIDENCE_STYLE[evidenceState] ?? EVIDENCE_STYLE.grounded;
  const latencyText = resp
    ? resp.latency_ms < 1000
      ? `${Math.round(resp.latency_ms)}ms`
      : `${(resp.latency_ms / 1000).toFixed(1)}s`
    : "";

  const officialSites =
    resp?.official_sites && resp.official_sites.length > 0
      ? resp.official_sites
      : evidenceState === "partial_missing"
        ? FALLBACK_OFFICIAL_SITES.public_housing
        : [];

  const followUps = (resp?.follow_ups || []).filter(Boolean).slice(0, 3);
  const followUpHeading =
    evidenceState === "partial_missing"
      ? "대신 바로 확인할 수 있는 질문"
      : evidenceState === "no_evidence"
        ? "확인 가능한 다른 질문"
        : "이어서 물어보기";

  const personalCheckQuery = `내 조건(${userProfile.household}, 월소득 ${userProfile.monthlyIncome}, ${userProfile.housingStatus}, 총자산 ${userProfile.assets})으로 방금 안내한 공고에 신청 가능한지 공고문 기준과 비교해서 알려줘.`;

  return (
    <div className="space-y-4">
      {/* User message */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        className="flex justify-end"
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-zinc-900 px-4 py-2.5 text-[14.5px] leading-relaxed text-white">
          {turn.question}
        </div>
      </motion.div>

      {/* Loading state */}
      {turn.isLoading ? (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-3.5"
        >
          <div className="flex w-14 shrink-0 flex-col items-center sm:w-16">
            <div className="flex h-14 w-14 items-end justify-center overflow-hidden rounded-2xl border border-[#E7E7E2] bg-white p-1 sm:h-16 sm:w-16">
              <img
                src={character.thinkingAvatar || character.avatar}
                alt={character.name}
                className="h-full w-full object-contain"
              />
            </div>
            <span className="mt-1.5 text-[11px] font-bold text-zinc-500">
              {character.name}
            </span>
          </div>

          <div className="bubble-tail-left relative flex-1 rounded-2xl border border-[#E7E7E2] bg-white px-5 py-4">
            <InvestigationStepper characterName={character.name} />
          </div>
        </motion.div>
      ) : resp ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.22 }}
          className="flex items-start gap-3.5"
        >
          {/* Left Character Portrait */}
          <div className="flex w-14 shrink-0 flex-col items-center sm:w-16">
            <div className="flex h-14 w-14 items-end justify-center overflow-hidden rounded-2xl border border-[#E7E7E2] bg-white p-1 sm:h-16 sm:w-16">
              <img
                src={portraitSrc}
                alt={character.name}
                className="h-full w-full object-contain"
              />
            </div>
            <span className="mt-1.5 text-[11.5px] font-bold text-zinc-800">
              {character.name}
            </span>
          </div>

          {/* Right Speech Bubble */}
          <div className="min-w-0 flex-1 space-y-3">
            <div className="bubble-tail-left relative rounded-2xl border border-[#E7E7E2] bg-white px-5 py-4 sm:px-6 sm:py-5">
              {/* Quiet Meta Header */}
              <div className="mb-3.5 flex items-center justify-between gap-2 border-b border-[#F2F2EE] pb-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className="inline-flex items-center gap-1.5 font-bold"
                    style={{ color: statusBadge.fg }}
                  >
                    <span
                      className="h-1.5 w-1.5 rounded-full"
                      style={{ backgroundColor: statusBadge.dot }}
                    />
                    {statusBadge.label}
                  </span>
                  <span className="text-zinc-300">·</span>
                  <span className="font-medium text-zinc-600">
                    {DOMAIN_LABELS[domain] || domain}
                  </span>
                  <span className="text-zinc-300">·</span>
                  <span className="font-mono text-[11px] text-zinc-400">
                    {latencyText}
                  </span>
                  {resp.meta?.reused_context && (
                    <span className="text-[11px] font-medium text-zinc-400">
                      · 직전 근거 활용
                    </span>
                  )}
                </div>

                {evidenceState === "grounded" && (
                  <button
                    type="button"
                    onClick={() => onToggleBookmark(turn)}
                    className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11.5px] font-medium transition ${
                      isBookmarked
                        ? "bg-zinc-900 text-white"
                        : "text-zinc-400 hover:bg-zinc-100 hover:text-zinc-800"
                    }`}
                    title="수첩에 스크랩"
                  >
                    {isBookmarked ? (
                      <>
                        <BookmarkCheck className="h-3.5 w-3.5" />
                        저장됨
                      </>
                    ) : (
                      <>
                        <Bookmark className="h-3.5 w-3.5" />
                        저장
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Main Text */}
              <div className="holmes-prose">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {displayedAnswer}
                </ReactMarkdown>
              </div>

              {/* Subtle 1-Click Actions (clean, understated text buttons) */}
              {isLatest && evidenceState === "grounded" && (
                <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-[#F2F2EE] pt-3">
                  <button
                    type="button"
                    onClick={() => onFollowUpClick(personalCheckQuery)}
                    className="rounded-lg border border-zinc-900 bg-zinc-900 px-2.5 py-1 text-[11.5px] font-semibold text-white transition hover:bg-zinc-700"
                  >
                    내 조건({userProfile.monthlyIncome})으로 신청 가능할까?
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      onFollowUpClick("방금 답변 핵심만 3줄로 깔끔하게 요약해줘.")
                    }
                    className="rounded-lg border border-[#E7E7E2] bg-[#FAFAF8] px-2.5 py-1 text-[11.5px] font-medium text-zinc-600 transition hover:border-zinc-400 hover:text-zinc-900"
                  >
                    3줄 요약
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      onFollowUpClick(
                        "방금 설명해 준 조건과 금액을 표로 깔끔하게 정리해줘."
                      )
                    }
                    className="rounded-lg border border-[#E7E7E2] bg-[#FAFAF8] px-2.5 py-1 text-[11.5px] font-medium text-zinc-600 transition hover:border-zinc-400 hover:text-zinc-900"
                  >
                    표로 보기
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      onFollowUpClick(
                        "사회초년생 눈높이에서 용어를 쉽게 풀어서 다시 설명해줘."
                      )
                    }
                    className="rounded-lg border border-[#E7E7E2] bg-[#FAFAF8] px-2.5 py-1 text-[11.5px] font-medium text-zinc-600 transition hover:border-zinc-400 hover:text-zinc-900"
                  >
                    쉽게 풀어서
                  </button>
                </div>
              )}

              {/* Citations */}
              {resp.citations && resp.citations.length > 0 && (
                <div className="mt-4 border-t border-[#F2F2EE] pt-3.5">
                  <div className="mb-2 text-[11.5px] font-bold text-zinc-400">
                    참고 출처 ({resp.citations.length})
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {resp.citations.map((c, idx) => {
                      const title = c.title || c.chunk_id || "원문 출처";
                      const sub = [c.source, c.location].filter(Boolean).join(" · ");
                      const content = (
                        <>
                          <div className="flex items-center gap-2">
                            <span className="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded bg-zinc-200/80 text-[10px] font-bold text-zinc-700">
                              {idx + 1}
                            </span>
                            <span className="truncate text-[12px] font-semibold text-zinc-800">
                              {title}
                            </span>
                            {c.url && (
                              <ExternalLink className="ml-auto h-3 w-3 shrink-0 text-zinc-400" />
                            )}
                          </div>
                          <span className="truncate text-[11px] text-zinc-400">
                            {sub || "원문 참조"}
                          </span>
                        </>
                      );
                      return c.url ? (
                        <a
                          key={idx}
                          href={c.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex flex-col gap-1 rounded-xl border border-[#ECECE8] bg-[#FAFAF8] p-2.5 transition hover:border-zinc-400 hover:bg-white"
                        >
                          {content}
                        </a>
                      ) : (
                        <div
                          key={idx}
                          className="flex flex-col gap-1 rounded-xl border border-[#ECECE8] bg-[#FAFAF8] p-2.5"
                        >
                          {content}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Case 2: Partial Missing -> Official Sites */}
              {evidenceState === "partial_missing" && (
                <div className="mt-4 space-y-3 border-t border-[#F2F2EE] pt-3.5">
                  <div className="rounded-xl border border-amber-200/80 bg-[#FFFBF2] px-3.5 py-2.5">
                    <div className="text-[12.5px] font-bold text-amber-950">
                      공고문 원문에 해당 세부 정보가 부족합니다
                    </div>
                    <p className="mt-0.5 text-[12px] text-amber-900/75">
                      추측하지 않고 직접 확인하실 수 있는 공식 사이트와 대체 질문을 함께 안내해 드립니다.
                    </p>
                  </div>

                  {officialSites.length > 0 && (
                    <div>
                      <div className="mb-2 text-[11.5px] font-bold text-zinc-400">
                        관련 공식 사이트
                      </div>
                      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                        {officialSites.map((site, idx) => (
                          <a
                            key={idx}
                            href={site.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group flex flex-col gap-0.5 rounded-xl border border-[#E7E7E2] bg-white p-3 transition hover:border-zinc-400"
                          >
                            <div className="flex items-center justify-between gap-2 text-[12.5px] font-bold text-zinc-900">
                              <span>{site.name}</span>
                              <ExternalLink className="h-3.5 w-3.5 shrink-0 text-zinc-400 group-hover:text-zinc-800" />
                            </div>
                            <p className="text-[11.5px] leading-snug text-zinc-500">
                              {site.desc}
                            </p>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Case 3: No Evidence */}
              {evidenceState === "no_evidence" && (
                <div className="mt-4 rounded-xl border border-[#E7E7E2] bg-[#F8F8F6] px-3.5 py-2.5">
                  <div className="text-[12.5px] font-bold text-zinc-900">
                    수집된 문서에 근거가 없어 사실대로 모른다고 안내드립니다
                  </div>
                  <p className="mt-0.5 text-[12px] text-zinc-500">
                    확인되지 않은 내용은 지어내지 않습니다. 아래에서 현재 서울시 데이터로 확인 가능한 질문을 골라 보세요.
                  </p>
                </div>
              )}
            </div>

            {/* Follow-up / Alternative Questions */}
            {isLatest && followUps.length > 0 && (
              <div className="space-y-1.5">
                <div className="pl-1 text-[11.5px] font-bold text-zinc-400">
                  {followUpHeading}
                </div>
                <div className="flex flex-col gap-1.5">
                  {followUps.map((fq, fIdx) => (
                    <button
                      key={fIdx}
                      type="button"
                      onClick={() => onFollowUpClick(fq)}
                      className="group flex items-center gap-2 rounded-xl border border-[#E7E7E2] bg-white px-3.5 py-2.5 text-left text-[13px] font-medium text-zinc-700 transition hover:border-zinc-400 hover:text-zinc-950"
                    >
                      <CornerDownRight className="h-3.5 w-3.5 shrink-0 text-zinc-400 group-hover:text-zinc-700" />
                      <span>{fq}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Collapsible Evidence Accordion */}
            <div className="overflow-hidden rounded-xl border border-[#E7E7E2] bg-white">
              <button
                type="button"
                onClick={() => setDetailsOpen((prev) => !prev)}
                className="flex w-full items-center justify-between px-3.5 py-2 text-left text-[12px] font-medium text-zinc-500 transition hover:bg-[#FAFAF8]"
              >
                <span className="inline-flex items-center gap-1.5">
                  <FileText className="h-3.5 w-3.5 text-zinc-400" />
                  검색 근거 및 라우팅 상세
                </span>
                {detailsOpen ? (
                  <ChevronUp className="h-4 w-4 text-zinc-400" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-zinc-400" />
                )}
              </button>

              {detailsOpen && (
                <div className="space-y-3 border-t border-[#F0F0EC] bg-[#FAFAF8] p-4 text-xs text-zinc-700">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <div className="font-bold text-zinc-500">라우팅 사유</div>
                      <div className="mt-0.5 text-zinc-800">
                        {resp.route_reason || "-"}
                      </div>
                    </div>
                    <div>
                      <div className="font-bold text-zinc-500">컬렉션</div>
                      <code className="mt-0.5 inline-block rounded bg-zinc-200/70 px-1.5 py-0.5 font-mono text-[11px] text-zinc-800">
                        {resp.collection_name || "-"}
                      </code>
                    </div>
                  </div>

                  {resp.resolved_query &&
                    resp.resolved_query.trim() !== turn.question.trim() && (
                      <div>
                        <div className="font-bold text-zinc-500">
                          맥락 반영 검색 질의
                        </div>
                        <div className="mt-1 rounded-lg border border-[#E7E7E2] bg-white px-2.5 py-1.5 font-medium text-zinc-800">
                          {resp.resolved_query}
                        </div>
                      </div>
                    )}

                  {resp.raw_answer &&
                    resp.raw_answer.trim() !== resp.answer.trim() && (
                      <div>
                        <div className="font-bold text-zinc-500">
                          원본 RAG 검색 답변
                        </div>
                        <div className="holmes-prose mt-1 rounded-lg border border-[#E7E7E2] bg-white p-3 text-[12.5px]">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {resp.raw_answer}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}

                  {resp.error_message && (
                    <div>
                      <div className="font-bold text-red-600">오류 상세</div>
                      <pre className="mt-1 overflow-x-auto rounded-lg bg-red-50 p-2.5 text-[11px] text-red-700">
                        {resp.error_message}
                      </pre>
                    </div>
                  )}

                  {resp.chunk_ids && resp.chunk_ids.length > 0 && (
                    <div>
                      <div className="font-bold text-zinc-500">검색된 청크 ID</div>
                      <pre className="mt-1 overflow-x-auto rounded-lg bg-zinc-900 p-2.5 text-[11px] text-zinc-100">
                        {resp.chunk_ids.join("\n")}
                      </pre>
                    </div>
                  )}

                  {resp.retrieved_contexts && resp.retrieved_contexts.length > 0 && (
                    <div className="space-y-2">
                      {resp.retrieved_contexts.slice(0, 4).map((ctx, idx) => (
                        <div key={idx}>
                          <div className="mb-1 font-bold text-zinc-500">
                            검색 근거 [{idx + 1}]
                          </div>
                          <div className="holmes-prose max-h-60 overflow-y-auto rounded-lg border border-[#E7E7E2] bg-white p-3 text-[12px]">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {ctx.length > 1500 ? `${ctx.slice(0, 1500)}\n\n…` : ctx}
                            </ReactMarkdown>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      ) : null}
    </div>
  );
}
