"use client";

import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Send } from "lucide-react";
import { DOMAIN_CONFIGS, DOMAIN_ORDER } from "@/lib/domains";
import type { DomainKey, HealthPayload } from "@/lib/types";

interface SeoulHeroIntroProps {
  health: HealthPayload | null;
  onStartWithQuestion: (question: string, part: DomainKey) => void;
  onSelectCategory: (part: DomainKey) => void;
}

interface CompanionHotspot {
  id: string;
  part: DomainKey;
  leftClass: string;
}

// Quiet invisible click zones aligned over the 4 companions in seoul_skyline.jpg (zero floating nameplates!)
const COMPANION_CLICK_ZONES: CompanionHotspot[] = [
  { id: "agasa", part: "rent_stats", leftClass: "left-[4%] w-[16%]" },
  { id: "haibara", part: "law", leftClass: "left-[20%] w-[16%]" },
  { id: "ran", part: "rent_deals", leftClass: "right-[14%] w-[16%]" },
  { id: "ayumi", part: "public_housing", leftClass: "right-[2%] w-[12%]" },
];

export function SeoulHeroIntro({
  health,
  onStartWithQuestion,
  onSelectCategory,
}: SeoulHeroIntroProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isPinned, setIsPinned] = useState(false);
  const [previewPart, setPreviewPart] = useState<DomainKey>("public_housing");
  const [quickInput, setQuickInput] = useState("");
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const isRevealed = isHovered || isPinned;
  const activeCfg = DOMAIN_CONFIGS[previewPart];

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const nx = (e.clientX / window.innerWidth - 0.5) * 8;
    const ny = (e.clientY / window.innerHeight - 0.5) * 6;
    setTilt({ x: -nx, y: -ny });
  };

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = quickInput.trim();
    if (!q) return;
    onStartWithQuestion(q, previewPart);
  };

  return (
    <div
      onMouseMove={handleMouseMove}
      className="relative h-screen w-full overflow-hidden bg-zinc-950 text-white select-none"
    >
      {/* Full-bleed Seoul Skyline Backdrop (16:9 aligned to bottom terrace) */}
      <div className="absolute inset-0 overflow-hidden">
        <img
          src="/scenes/seoul_skyline.jpg?v=2"
          alt="서울 야경을 바라보며 집을 찾는 구해줘 홈즈 등장인물들의 뒷모습"
          className="h-full w-full object-cover object-bottom transition-transform duration-700 ease-out"
          style={{
            transform: `translate3d(${tilt.x}px, ${tilt.y}px, 0) scale(${
              isRevealed ? 1.04 : 1.015
            })`,
          }}
        />
        <div
          className="absolute inset-0 transition-opacity duration-500"
          style={{
            background: isRevealed
              ? "linear-gradient(to bottom, rgba(0,0,0,0.58) 0%, rgba(0,0,0,0.28) 45%, rgba(0,0,0,0.72) 100%)"
              : "linear-gradient(to bottom, rgba(0,0,0,0.52) 0%, rgba(0,0,0,0.12) 48%, rgba(0,0,0,0.25) 100%)",
          }}
        />
      </div>

      {/* Minimal Top Bar */}
      <header className="relative z-30 mx-auto flex max-w-6xl items-center justify-between px-6 py-5 sm:px-8">
        <div className="flex items-center gap-2.5">
          <span className="font-display text-[19px] font-extrabold tracking-tight text-white">
            구해줘 홈즈
          </span>
          <span className="text-white/30">·</span>
          <span className="text-xs font-medium text-white/75">
            서울 주거 탐색
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-1.5 text-xs text-white/80 sm:inline-flex">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health?.ok ? "bg-emerald-400" : "bg-amber-300"
              }`}
            />
            {health?.ok
              ? `${health.registered_domains.length}개 데이터 연결`
              : "연결 확인 중"}
          </span>
          <button
            type="button"
            onClick={() => onSelectCategory("public_housing")}
            className="inline-flex items-center gap-1.5 rounded-full bg-white px-4 py-2 text-xs font-bold text-zinc-900 shadow-sm transition hover:bg-zinc-100"
          >
            바로 상담하기
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </header>

      {/* Editorial Hero Title (clean upper sky area) */}
      <motion.div
        animate={{
          y: isRevealed ? -18 : 0,
          opacity: isRevealed ? 0.85 : 1,
        }}
        transition={{ duration: 0.3 }}
        className="relative z-10 mx-auto mt-1 max-w-2xl px-6 text-center sm:mt-3"
      >
        <p className="mb-2 text-xs font-semibold tracking-wide text-white/75">
          공공임대 공고 · 임대차 법령 · 전월세 조건 · 서울 25구 시세
        </p>
        <h1 className="font-display text-3xl font-extrabold leading-[1.22] tracking-tight text-white drop-shadow sm:text-[46px]">
          서울에서 내 집 찾기,
          <br />
          구해줘 홈즈
        </h1>
        <p className="mx-auto mt-2.5 max-w-md text-sm leading-relaxed text-white/80">
          {isRevealed
            ? "궁금한 카테고리나 질문을 고르면 바로 대화가 시작돼요."
            : "가운데 코난 뒷모습에 마우스를 가져가 보세요."}
        </p>
      </motion.div>

      {/* Clean Invisible Click Zones over the 4 Companions (no cluttered nameplates) */}
      {!isRevealed &&
        COMPANION_CLICK_ZONES.map((comp) => (
          <button
            key={comp.id}
            type="button"
            onClick={() => onSelectCategory(comp.part)}
            aria-label={`${DOMAIN_CONFIGS[comp.part].label} 대화방 열기`}
            className={`absolute bottom-[6%] z-20 hidden h-[44vh] cursor-pointer bg-transparent md:block ${comp.leftClass}`}
          />
        ))}

      {/* Interactive Conan Standing Full-Scale Right on the Terrace Floor */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex flex-col items-center justify-end"
        style={{ height: "78vh" }}
      >
        <div
          onMouseLeave={() => {
            if (!isPinned) setIsHovered(false);
          }}
          className="pointer-events-auto relative flex w-full max-w-4xl flex-col items-center justify-end px-4"
        >
          {/* Default State: Full-scale Conan Back-View aligned with Haibara & Ran on the terrace */}
          <AnimatePresence mode="wait">
            {!isRevealed ? (
              <motion.div
                key="conan-terrace-back"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0.16 } }}
                onMouseEnter={() => setIsHovered(true)}
                onClick={() => setIsPinned(true)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setIsPinned(true);
                }}
                aria-label="코난 뒷모습에 마우스를 올려 첫 질문 열기"
                className="relative mb-[6.8vh] flex cursor-pointer flex-col items-center focus:outline-none"
              >
                <motion.div
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 2.1, repeat: Infinity, ease: "easeInOut" }}
                  className="mb-2 rounded-full bg-white/95 px-3.5 py-1.5 text-xs font-bold text-zinc-900 shadow-lg"
                >
                  코난에게 마우스를 올려보세요
                </motion.div>

                <motion.img
                  src="/characters/conan_back.png"
                  alt="서울 야경을 바라보는 코난의 뒷모습"
                  initial={{ rotateY: 50, scale: 0.96 }}
                  animate={{ rotateY: 0, scale: 1 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                  className="h-[38.5vh] max-h-[395px] min-h-[240px] w-auto object-contain drop-shadow-[0_18px_28px_rgba(0,0,0,0.6)]"
                />
              </motion.div>
            ) : (
              /* Revealed State: Conan Turns Around Full-Size Right Behind the Starter Question Sheet */
              <motion.div
                key="conan-terrace-front"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0.16 } }}
                onMouseEnter={() => setIsHovered(true)}
                className="flex w-full flex-col items-center pb-6"
              >
                {/* Conan Front + Speech Bubble Row */}
                <div className="relative z-10 -mb-5 flex flex-col items-center sm:flex-row sm:items-center sm:gap-4">
                  <motion.img
                    src="/characters/conan_front.png"
                    alt="뒤돌아 바라보는 코난의 앞모습"
                    onClick={() => setIsPinned((prev) => !prev)}
                    initial={{ opacity: 0, rotateY: 85, scale: 0.9 }}
                    animate={{ opacity: 1, rotateY: 0, scale: 1 }}
                    transition={{ type: "spring", stiffness: 260, damping: 22 }}
                    className="h-[30vh] max-h-[300px] min-h-[200px] w-auto cursor-pointer object-contain drop-shadow-[0_20px_28px_rgba(0,0,0,0.5)]"
                  />

                  <motion.div
                    initial={{ opacity: 0, x: -10, scale: 0.96 }}
                    animate={{ opacity: 1, x: 0, scale: 1 }}
                    transition={{ duration: 0.22, delay: 0.05 }}
                    className="relative mb-6 max-w-xs rounded-2xl bg-white px-5 py-3.5 text-left text-zinc-900 shadow-2xl sm:mb-8 sm:max-w-sm"
                  >
                    <p className="text-[14.5px] font-bold leading-snug text-zinc-900">
                      어, 서울에서 집 찾고 있구나!
                      <br />
                      구해줘 홈즈, 나랑 친구들한테 맡겨봐.
                    </p>
                    <span className="absolute -left-1.5 top-1/2 hidden h-3 w-3 -translate-y-1/2 rotate-45 bg-white sm:block" />
                  </motion.div>
                </div>

                {/* Clean Starter Question Sheet */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.24 }}
                  className="relative z-20 w-full rounded-2xl border border-black/10 bg-white/95 p-4 text-zinc-900 shadow-2xl backdrop-blur-md sm:p-5"
                >
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-zinc-100 pb-3">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {DOMAIN_ORDER.map((key) => {
                        const cfg = DOMAIN_CONFIGS[key];
                        const active = previewPart === key;
                        return (
                          <button
                            key={key}
                            type="button"
                            onClick={() => setPreviewPart(key)}
                            className={`rounded-full px-3 py-1 text-xs font-bold transition ${
                              active
                                ? "bg-zinc-900 text-white"
                                : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                            }`}
                          >
                            {cfg.label}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      type="button"
                      onClick={() => onSelectCategory(previewPart)}
                      className="inline-flex items-center gap-1 text-xs font-bold text-zinc-700 hover:text-zinc-950"
                    >
                      {activeCfg.label} 대화방으로
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {activeCfg.samples.map((item, idx) => (
                      <button
                        key={`${previewPart}-${idx}`}
                        type="button"
                        onClick={() => onStartWithQuestion(item.question, previewPart)}
                        className="group flex items-start justify-between gap-2 rounded-xl border border-zinc-200 bg-[#FAFAF8] px-3.5 py-2.5 text-left transition hover:border-zinc-400 hover:bg-white"
                      >
                        <div className="min-w-0">
                          <div className="text-[12.5px] font-bold text-zinc-900">
                            {item.label}
                          </div>
                          <p className="mt-0.5 line-clamp-1 text-[11.5px] text-zinc-500">
                            {item.question}
                          </p>
                        </div>
                        <ArrowRight className="mt-1 h-3.5 w-3.5 shrink-0 text-zinc-300 transition group-hover:text-zinc-700" />
                      </button>
                    ))}
                  </div>

                  <form
                    onSubmit={handleQuickSubmit}
                    className="mt-3 flex items-center gap-2 rounded-xl border border-zinc-300 bg-white px-3.5 py-2 focus-within:border-zinc-900"
                  >
                    <input
                      type="text"
                      value={quickInput}
                      onChange={(e) => setQuickInput(e.target.value)}
                      placeholder={activeCfg.placeholder}
                      className="w-full bg-transparent text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none"
                    />
                    <button
                      type="submit"
                      disabled={!quickInput.trim()}
                      className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-900 text-white transition hover:bg-zinc-700 disabled:bg-zinc-200 disabled:text-zinc-400"
                    >
                      <Send className="h-3.5 w-3.5" />
                    </button>
                  </form>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
