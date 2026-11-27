"use client";

import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bookmark,
  ChevronDown,
  ChevronUp,
  History,
  Home,
  Menu,
  MessageSquarePlus,
  Send,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import {
  CHARACTERS,
  DOMAIN_CONFIGS,
  DOMAIN_LABELS,
  DOMAIN_ORDER,
} from "@/lib/domains";
import type {
  ChatTurn,
  DomainKey,
  HealthPayload,
  SavedClue,
  UserHousingProfile,
} from "@/lib/types";
import { ChatBubble } from "./ChatBubble";
import { SampleQuestionGrid } from "./SampleQuestionGrid";

interface ChatWorkspaceProps {
  selectedPart: DomainKey;
  onSelectPart: (part: DomainKey) => void;
  turns: ChatTurn[];
  health: HealthPayload | null;
  conversationalMode: boolean;
  onToggleConversational: (val: boolean) => void;
  userProfile: UserHousingProfile;
  onChangeProfile: (next: UserHousingProfile) => void;
  savedClues: SavedClue[];
  onToggleBookmark: (turn: ChatTurn) => void;
  onRemoveClue: (id: string) => void;
  onSendQuestion: (question: string) => void;
  onNewChat: () => void;
  onBackToHero: () => void;
}

const HOUSEHOLD_OPTIONS = ["1인 청년", "신혼부부", "한부모가족"];
const INCOME_OPTIONS = ["250만원", "350만원", "450만원"];
const ASSET_OPTIONS = ["1.5억 이하", "2.5억 이하", "3.5억 이하"];

export function ChatWorkspace({
  selectedPart,
  onSelectPart,
  turns,
  health,
  conversationalMode,
  onToggleConversational,
  userProfile,
  onChangeProfile,
  savedClues,
  onToggleBookmark,
  onRemoveClue,
  onSendQuestion,
  onNewChat,
  onBackToHero,
}: ChatWorkspaceProps) {
  const [input, setInput] = useState("");
  const [filterByPart, setFilterByPart] = useState(false);
  const [samplesExpanded, setSamplesExpanded] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<"history" | "clues">("history");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const cfg = DOMAIN_CONFIGS[selectedPart];
  const activeChar = CHARACTERS[cfg.characterId];
  const companionChar = cfg.companionId ? CHARACTERS[cfg.companionId] : null;
  const registeredSet = new Set(health?.registered_domains || []);
  const savedIds = new Set(savedClues.map((c) => c.id));

  const visibleTurns = turns.filter((t) => {
    if (!filterByPart || selectedPart === "auto" || !t.response) return true;
    return (
      t.response.domain === selectedPart || t.selectedPart === selectedPart
    );
  });

  const recentQuestions = Array.from(
    new Set([...turns].reverse().map((t) => t.question))
  ).slice(0, 6);

  const isBusy = turns.some((t) => t.isLoading);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, isBusy]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || isBusy) return;
    setInput("");
    onSendQuestion(q);
  };

  const sidebarContent = (
    <div className="flex h-full flex-col bg-[#FBFBF9] px-4 py-5 text-zinc-900">
      {/* Clean Editorial Brand */}
      <div className="mb-4 flex items-center justify-between border-b border-[#ECECE8] pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-[#E5E5E0] bg-white p-0.5">
            <img
              src="/characters/conan_front.png"
              alt="구해줘 홈즈"
              className="h-full w-full object-contain"
            />
          </div>
          <div>
            <div className="font-display text-[16px] font-extrabold tracking-tight text-zinc-900">
              구해줘 홈즈
            </div>
            <div className="text-[11px] text-zinc-400">서울 주거 탐색</div>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setMobileMenuOpen(false)}
          className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-200/60 lg:hidden"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Primary Actions */}
      <div className="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            setMobileMenuOpen(false);
          }}
          className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-zinc-900 px-3 py-2 text-xs font-bold text-white transition hover:bg-zinc-800"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />새 대화
        </button>
        <button
          type="button"
          onClick={onBackToHero}
          className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-[#E5E5E0] bg-white px-3 py-2 text-xs font-semibold text-zinc-700 transition hover:border-zinc-400"
        >
          <Home className="h-3.5 w-3.5" />
          인트로
        </button>
      </div>

      {/* Domain Cast List */}
      <div className="mt-5">
        <div className="mb-2 px-1 text-[11px] font-semibold text-zinc-400">
          카테고리별 담당
        </div>
        <div className="space-y-1">
          {(["public_housing", "law", "rent_deals", "rent_stats"] as DomainKey[]).map(
            (key) => {
              const item = DOMAIN_CONFIGS[key];
              const ch = CHARACTERS[item.characterId];
              const active = selectedPart === key;
              const online = registeredSet.has(key);
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => {
                    onSelectPart(active ? "auto" : key);
                    setMobileMenuOpen(false);
                  }}
                  className={`flex w-full items-center justify-between rounded-xl px-2.5 py-2 text-left transition ${
                    active
                      ? "bg-white text-zinc-950 shadow-sm ring-1 ring-[#E2E2DC]"
                      : "text-zinc-600 hover:bg-[#F2F2EE] hover:text-zinc-900"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <div className="relative h-7 w-7 shrink-0 overflow-hidden rounded-full border border-[#E5E5E0] bg-white p-0.5">
                      <img
                        src={ch.avatar}
                        alt={ch.name}
                        className="h-full w-full object-contain"
                      />
                      <span
                        className={`absolute bottom-0 right-0 h-2 w-2 rounded-full ring-1 ring-white ${
                          online ? "bg-emerald-500" : "bg-zinc-300"
                        }`}
                      />
                    </div>
                    <div>
                      <div className="text-[12.5px] font-bold leading-tight">
                        {item.label}
                      </div>
                      <div className="text-[10.5px] text-zinc-400">
                        {ch.name} · {item.owner.split(" ")[0]}
                      </div>
                    </div>
                  </div>
                </button>
              );
            }
          )}
        </div>
      </div>

      {/* History / Saved Notebook */}
      <div className="mt-5 flex min-h-0 flex-1 flex-col">
        <div className="mb-2 flex items-center gap-3 border-b border-[#ECECE8] px-1 pb-1.5 text-[11.5px] font-bold">
          <button
            type="button"
            onClick={() => setSidebarTab("history")}
            className={`inline-flex items-center gap-1 pb-1 transition ${
              sidebarTab === "history"
                ? "border-b-2 border-zinc-900 text-zinc-900"
                : "text-zinc-400 hover:text-zinc-700"
            }`}
          >
            <History className="h-3 w-3" />
            최근 질문
          </button>
          <button
            type="button"
            onClick={() => setSidebarTab("clues")}
            className={`inline-flex items-center gap-1 pb-1 transition ${
              sidebarTab === "clues"
                ? "border-b-2 border-zinc-900 text-zinc-900"
                : "text-zinc-400 hover:text-zinc-700"
            }`}
          >
            <Bookmark className="h-3 w-3" />
            저장 ({savedClues.length})
          </button>
        </div>

        <div className="flex-1 overflow-y-auto pr-0.5">
          {sidebarTab === "history" ? (
            recentQuestions.length > 0 ? (
              <div className="space-y-0.5">
                {recentQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    disabled={isBusy}
                    onClick={() => {
                      onSendQuestion(q);
                      setMobileMenuOpen(false);
                    }}
                    title={q}
                    className="w-full truncate rounded-lg px-2 py-1.5 text-left text-[12px] text-zinc-600 transition hover:bg-[#F0F0EC] hover:text-zinc-950 disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : (
              <p className="px-1 text-[11.5px] text-zinc-400">
                대화한 질문이 여기에 쌓입니다.
              </p>
            )
          ) : savedClues.length > 0 ? (
            <div className="space-y-2">
              {savedClues.map((clue) => (
                <div
                  key={clue.id}
                  className="rounded-xl border border-[#E7E7E2] bg-white p-2.5"
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-[10.5px] font-bold text-zinc-500">
                      {clue.characterName} · {DOMAIN_LABELS[clue.domain] || clue.domain}
                    </span>
                    <button
                      type="button"
                      onClick={() => onRemoveClue(clue.id)}
                      className="text-zinc-400 hover:text-red-500"
                      title="삭제"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  <p className="mt-1 line-clamp-2 text-[11.5px] font-semibold text-zinc-800">
                    {clue.question}
                  </p>
                  <p className="mt-1 line-clamp-2 text-[11px] text-zinc-500">
                    {clue.summary}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="px-1 text-[11.5px] leading-relaxed text-zinc-400">
              답변의 <strong className="text-zinc-600">저장</strong> 버튼을 누르면 중요한 공고 조건을 모아둘 수 있어요.
            </p>
          )}
        </div>
      </div>

      {/* Footer Toggles */}
      <div className="mt-4 space-y-2.5 border-t border-[#ECECE8] pt-4 text-xs">
        <label className="flex cursor-pointer items-center justify-between text-zinc-700">
          <span className="font-medium">대화형 상담 모드</span>
          <input
            type="checkbox"
            checked={conversationalMode}
            onChange={(e) => onToggleConversational(e.target.checked)}
            className="h-3.5 w-3.5 accent-zinc-900"
          />
        </label>
        <label
          className={`flex items-center justify-between ${
            selectedPart === "auto"
              ? "cursor-not-allowed text-zinc-400"
              : "cursor-pointer text-zinc-700"
          }`}
        >
          <span className="font-medium">선택 도메인만 보기</span>
          <input
            type="checkbox"
            disabled={selectedPart === "auto"}
            checked={filterByPart && selectedPart !== "auto"}
            onChange={(e) => setFilterByPart(e.target.checked)}
            className="h-3.5 w-3.5 accent-zinc-900"
          />
        </label>

        <div className="pt-1 text-[11px] text-zinc-400">
          {health?.ok
            ? `${health.registered_domains.length}/${health.total_domains} 도메인 연결됨`
            : "백엔드(8000) 연결 확인 필요"}
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen bg-[#F8F8F6]">
      {/* Desktop Light Editorial Sidebar */}
      <aside className="hidden w-68 shrink-0 border-r border-[#E7E7E2] lg:block" style={{ width: 268 }}>
        <div className="sticky top-0 h-screen">{sidebarContent}</div>
      </aside>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileMenuOpen(false)}
              className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm lg:hidden"
            />
            <motion.aside
              initial={{ x: -272 }}
              animate={{ x: 0 }}
              exit={{ x: -272 }}
              transition={{ type: "spring", stiffness: 320, damping: 32 }}
              className="fixed inset-y-0 left-0 z-50 w-68 shadow-2xl lg:hidden"
            >
              {sidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main Column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Clean Minimal Header */}
        <header className="sticky top-0 z-30 border-b border-[#E7E7E2] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-3 sm:px-6">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(true)}
                className="rounded-lg border border-[#E5E5E0] p-1.5 text-zinc-600 hover:bg-zinc-100 lg:hidden"
              >
                <Menu className="h-4 w-4" />
              </button>

              <div className="flex flex-1 items-center gap-1 overflow-x-auto rounded-xl bg-[#F3F3EF] p-1">
                {DOMAIN_ORDER.map((key) => {
                  const item = DOMAIN_CONFIGS[key];
                  const active = selectedPart === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onSelectPart(key)}
                      className={`relative flex-1 whitespace-nowrap rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                        active ? "text-zinc-950" : "text-zinc-500 hover:text-zinc-900"
                      }`}
                    >
                      {active && (
                        <motion.span
                          layoutId="active-domain-pill"
                          transition={{ type: "spring", stiffness: 420, damping: 34 }}
                          className="absolute inset-0 rounded-lg bg-white shadow-sm ring-1 ring-black/5"
                        />
                      )}
                      <span className="relative z-10">{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-[11.5px]">
              <span className="font-medium text-zinc-500">
                {cfg.fullName} · {cfg.owner}
              </span>
              <button
                type="button"
                onClick={() => setProfileOpen((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold transition ${
                  profileOpen
                    ? "border-zinc-900 bg-zinc-900 text-white"
                    : "border-[#E5E5E0] bg-[#FAFAF8] text-zinc-600 hover:border-zinc-400"
                }`}
              >
                <SlidersHorizontal className="h-3 w-3" />
                내 조건 · {userProfile.household} / 월 {userProfile.monthlyIncome}
              </button>
            </div>

            {/* Personal Profile Selector */}
            <AnimatePresence>
              {profileOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-1 grid grid-cols-1 gap-3 rounded-xl border border-[#E7E7E2] bg-[#FAFAF8] p-3 text-xs sm:grid-cols-3">
                    <div>
                      <div className="mb-1 text-[11px] font-semibold text-zinc-500">
                        가구 유형
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {HOUSEHOLD_OPTIONS.map((opt) => (
                          <button
                            key={opt}
                            type="button"
                            onClick={() =>
                              onChangeProfile({ ...userProfile, household: opt })
                            }
                            className={`rounded-md px-2 py-1 text-[11px] font-semibold transition ${
                              userProfile.household === opt
                                ? "bg-zinc-900 text-white"
                                : "border border-[#E5E5E0] bg-white text-zinc-600 hover:border-zinc-400"
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-[11px] font-semibold text-zinc-500">
                        월 평균 소득
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {INCOME_OPTIONS.map((opt) => (
                          <button
                            key={opt}
                            type="button"
                            onClick={() =>
                              onChangeProfile({ ...userProfile, monthlyIncome: opt })
                            }
                            className={`rounded-md px-2 py-1 text-[11px] font-semibold transition ${
                              userProfile.monthlyIncome === opt
                                ? "bg-zinc-900 text-white"
                                : "border border-[#E5E5E0] bg-white text-zinc-600 hover:border-zinc-400"
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-[11px] font-semibold text-zinc-500">
                        총자산 범위
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {ASSET_OPTIONS.map((opt) => (
                          <button
                            key={opt}
                            type="button"
                            onClick={() =>
                              onChangeProfile({ ...userProfile, assets: opt })
                            }
                            className={`rounded-md px-2 py-1 text-[11px] font-semibold transition ${
                              userProfile.assets === opt
                                ? "bg-zinc-900 text-white"
                                : "border border-[#E5E5E0] bg-white text-zinc-600 hover:border-zinc-400"
                            }`}
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </header>

        {/* Scrollable Dialogue Area */}
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-32 pt-6 sm:px-6">
          {/* Natural Character Intro Line (no nested boxes) */}
          <motion.section
            key={selectedPart}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="mb-6 flex items-center gap-4 border-b border-[#E7E7E2] pb-5"
          >
            <div className="relative flex shrink-0 items-end">
              <div className="flex h-20 w-20 items-end justify-center overflow-hidden rounded-2xl border border-[#E7E7E2] bg-white p-1.5">
                <img
                  src={activeChar.avatar}
                  alt={activeChar.name}
                  className="h-full w-full object-contain"
                />
              </div>
              {companionChar && (
                <div className="-ml-4 flex h-11 w-11 items-end justify-center overflow-hidden rounded-xl border border-[#E7E7E2] bg-white p-0.5 shadow-sm">
                  <img
                    src={companionChar.avatar}
                    alt={companionChar.name}
                    className="h-full w-full object-contain"
                  />
                </div>
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="text-xs font-bold text-zinc-500">
                {activeChar.name} · {activeChar.role}
              </div>
              <p className="mt-1 text-[14.5px] font-semibold leading-relaxed text-zinc-900">
                &ldquo;{cfg.greeting}&rdquo;
              </p>
            </div>
          </motion.section>

          {/* Sample Questions */}
          {visibleTurns.length === 0 ? (
            <section className="mb-8">
              <SampleQuestionGrid
                selectedPart={selectedPart}
                config={cfg}
                onSelectQuestion={onSendQuestion}
              />
            </section>
          ) : (
            <div className="mb-6 overflow-hidden rounded-xl border border-[#E7E7E2] bg-white">
              <button
                type="button"
                onClick={() => setSamplesExpanded((prev) => !prev)}
                className="flex w-full items-center justify-between px-4 py-2.5 text-left text-xs font-semibold text-zinc-600 transition hover:bg-[#FAFAF8]"
              >
                <span>
                  {cfg.label} 추천 질문 보기
                  {selectedPart === "public_housing"
                    ? " (근거 확인 · 근거 부족 · 근거 없음)"
                    : ""}
                </span>
                {samplesExpanded ? (
                  <ChevronUp className="h-4 w-4 text-zinc-400" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-zinc-400" />
                )}
              </button>
              <AnimatePresence>
                {samplesExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden border-t border-[#F0F0EC] p-4"
                  >
                    <SampleQuestionGrid
                      selectedPart={selectedPart}
                      config={cfg}
                      onSelectQuestion={(q) => {
                        setSamplesExpanded(false);
                        onSendQuestion(q);
                      }}
                      compact
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Conversation Feed */}
          <div className="space-y-8">
            {visibleTurns.map((turn, idx) => (
              <ChatBubble
                key={turn.id}
                turn={turn}
                isLatest={idx === visibleTurns.length - 1}
                conversationalMode={conversationalMode}
                userProfile={userProfile}
                isBookmarked={savedIds.has(turn.id)}
                onToggleBookmark={onToggleBookmark}
                onFollowUpClick={onSendQuestion}
              />
            ))}
          </div>

          <div ref={bottomRef} />
        </main>

        {/* Pinned Bottom Input */}
        <div className="sticky bottom-0 z-20 border-t border-[#E7E7E2] bg-white/95 px-4 py-3.5 backdrop-blur-md sm:px-6">
          <form
            onSubmit={handleSubmit}
            className="mx-auto flex max-w-3xl items-center gap-2.5 rounded-xl border border-[#D8D8D2] bg-white px-3.5 py-2 transition focus-within:border-zinc-900"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isBusy}
              placeholder={cfg.placeholder}
              className="w-full bg-transparent text-[14.5px] text-zinc-900 placeholder:text-zinc-400 focus:outline-none disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={!input.trim() || isBusy}
              className="inline-flex h-8 shrink-0 items-center gap-1 rounded-lg bg-zinc-900 px-3 text-xs font-bold text-white transition hover:bg-zinc-700 disabled:bg-zinc-200 disabled:text-zinc-400"
            >
              <span>전송</span>
              <Send className="h-3.5 w-3.5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
