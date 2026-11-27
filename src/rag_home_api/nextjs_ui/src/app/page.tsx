"use client";

import React, { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChatWorkspace } from "@/components/ChatWorkspace";
import { SeoulHeroIntro } from "@/components/SeoulHeroIntro";
import { CHARACTERS, getCharacterForDomain } from "@/lib/domains";
import type {
  ChatTurn,
  DomainKey,
  HealthPayload,
  RAGResponsePayload,
  SavedClue,
  UserHousingProfile,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_RAG_API_URL || "http://127.0.0.1:8000";

export default function HomePage() {
  const [view, setView] = useState<"hero" | "chat">("hero");
  const [selectedPart, setSelectedPart] = useState<DomainKey>("public_housing");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [conversationalMode, setConversationalMode] = useState(true);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [savedClues, setSavedClues] = useState<SavedClue[]>([]);
  const [userProfile, setUserProfile] = useState<UserHousingProfile>({
    household: "1인 청년",
    monthlyIncome: "350만원",
    assets: "2.5억 이하",
    housingStatus: "무주택",
  });

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: HealthPayload | null) => {
        if (active && data) setHealth(data);
      })
      .catch(() => {
        if (active) setHealth(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleToggleBookmark = useCallback((turn: ChatTurn) => {
    setSavedClues((prev) => {
      const exists = prev.some((item) => item.id === turn.id);
      if (exists) {
        return prev.filter((item) => item.id !== turn.id);
      }
      const resp = turn.response;
      const char =
        resp?.character_id && CHARACTERS[resp.character_id]
          ? CHARACTERS[resp.character_id]
          : getCharacterForDomain(resp?.domain, turn.selectedPart);
      const summary = (resp?.answer || "")
        .replace(/[#*`]/g, "")
        .replace(/\s+/g, " ")
        .slice(0, 120);
      return [
        {
          id: turn.id,
          question: turn.question,
          domain: resp?.domain || turn.selectedPart,
          characterName: char.name,
          summary,
          savedAt: new Date().toLocaleTimeString("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
        ...prev,
      ];
    });
  }, []);

  const handleRemoveClue = useCallback((id: string) => {
    setSavedClues((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const sendQuestion = useCallback(
    async (question: string, overridePart?: DomainKey) => {
      const part = overridePart ?? selectedPart;
      const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

      const historyPayload: Array<{
        role: "user" | "assistant";
        content: string;
        response?: RAGResponsePayload;
      }> = [];
      for (const t of turns) {
        if (t.response && !t.isLoading) {
          historyPayload.push({ role: "user", content: t.question });
          historyPayload.push({
            role: "assistant",
            content: t.response.answer,
            response: t.response,
          });
        }
      }

      setTurns((prev) => [
        ...prev.map((item) => ({ ...item, isNew: false })),
        {
          id: turnId,
          question,
          selectedPart: part,
          isLoading: true,
          isNew: true,
        },
      ]);

      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            selected_part: part,
            history: historyPayload,
            conversational_mode: conversationalMode,
          }),
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data: RAGResponsePayload = await res.json();
        setTurns((prev) =>
          prev.map((item) =>
            item.id === turnId
              ? { ...item, isLoading: false, response: data }
              : item
          )
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        const fallbackResp: RAGResponsePayload = {
          status: "error",
          domain: part === "auto" ? "unsupported" : part,
          selected_part: part,
          character_id: "conan",
          answer:
            "FastAPI 백엔드 서버(`http://127.0.0.1:8000`)에 연결하지 못했습니다. 백엔드 서버가 실행 중인지 확인해 주세요.",
          citations: [],
          retrieved_contexts: [],
          chunk_ids: [],
          latency_ms: 0,
          collection_name: "",
          route_reason: "백엔드 통신 오류",
          error_message: errMsg,
          meta: { evidence_state: "error" },
          follow_ups: [],
          official_sites: [],
        };
        setTurns((prev) =>
          prev.map((item) =>
            item.id === turnId
              ? { ...item, isLoading: false, response: fallbackResp }
              : item
          )
        );
      }
    },
    [selectedPart, turns, conversationalMode]
  );

  const handleHeroQuestion = useCallback(
    (question: string, part: DomainKey) => {
      setSelectedPart(part);
      setView("chat");
      void sendQuestion(question, part);
    },
    [sendQuestion]
  );

  const handleSelectCategory = useCallback((part: DomainKey) => {
    setSelectedPart(part);
    setView("chat");
  }, []);

  return (
    <AnimatePresence mode="wait">
      {view === "hero" ? (
        <motion.div
          key="hero-view"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 0.985 }}
          transition={{ duration: 0.3 }}
        >
          <SeoulHeroIntro
            health={health}
            onStartWithQuestion={handleHeroQuestion}
            onSelectCategory={handleSelectCategory}
          />
        </motion.div>
      ) : (
        <motion.div
          key="chat-view"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
        >
          <ChatWorkspace
            selectedPart={selectedPart}
            onSelectPart={setSelectedPart}
            turns={turns}
            health={health}
            conversationalMode={conversationalMode}
            onToggleConversational={setConversationalMode}
            userProfile={userProfile}
            onChangeProfile={setUserProfile}
            savedClues={savedClues}
            onToggleBookmark={handleToggleBookmark}
            onRemoveClue={handleRemoveClue}
            onSendQuestion={(q) => void sendQuestion(q)}
            onNewChat={() => setTurns([])}
            onBackToHero={() => setView("hero")}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
