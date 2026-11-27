"use client";

import React from "react";
import { ArrowUpRight } from "lucide-react";
import type { DomainConfig } from "@/lib/domains";
import type { DomainKey } from "@/lib/types";

interface SampleQuestionGridProps {
  selectedPart: DomainKey;
  config: DomainConfig;
  onSelectQuestion: (question: string) => void;
  compact?: boolean;
}

export function SampleQuestionGrid({
  config,
  onSelectQuestion,
  compact = false,
}: SampleQuestionGridProps) {
  if (config.sampleGroups && config.sampleGroups.length > 0) {
    return (
      <div className={compact ? "space-y-4" : "space-y-5"}>
        {config.sampleGroups.map((group, gIdx) => (
          <div
            key={group.type}
            className={gIdx > 0 ? "border-t border-[#ECECE8] pt-4" : ""}
          >
            <div className="mb-2.5 flex flex-wrap items-baseline gap-2">
              <span
                className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11.5px] font-bold"
                style={{
                  color: group.fg,
                  backgroundColor: group.bg,
                }}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: group.fg }}
                />
                {group.badge}
              </span>
              <span className="text-xs text-zinc-500">{group.desc}</span>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {group.items.map((item, idx) => (
                <button
                  key={`${group.type}-${idx}`}
                  type="button"
                  onClick={() => onSelectQuestion(item.question)}
                  className="group flex flex-col justify-between rounded-xl border border-[#E7E7E2] bg-white px-4 py-3 text-left transition hover:border-zinc-400 hover:bg-[#FAFAF8]"
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="text-[13px] font-bold text-zinc-900">
                      {item.label}
                    </span>
                    <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-zinc-300 transition group-hover:text-zinc-700" />
                  </div>
                  <p className="line-clamp-2 text-[12px] leading-relaxed text-zinc-500">
                    {item.question}
                  </p>
                </button>
              ))}
            </div>

            {group.hint && (
              <p className="mt-2 pl-0.5 text-[12px] text-zinc-500">
                팁: {group.hint}
              </p>
            )}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {config.samples.map((item, idx) => (
        <button
          key={idx}
          type="button"
          onClick={() => onSelectQuestion(item.question)}
          className="group flex flex-col justify-between rounded-xl border border-[#E7E7E2] bg-white px-4 py-3 text-left transition hover:border-zinc-400 hover:bg-[#FAFAF8]"
        >
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="text-[13px] font-bold text-zinc-900">
              {item.label}
            </span>
            <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-zinc-300 transition group-hover:text-zinc-700" />
          </div>
          <p className="line-clamp-2 text-[12px] leading-relaxed text-zinc-500">
            {item.question}
          </p>
        </button>
      ))}
    </div>
  );
}
