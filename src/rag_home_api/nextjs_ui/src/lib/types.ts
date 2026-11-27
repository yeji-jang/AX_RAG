export type DomainKey = "auto" | "public_housing" | "law" | "rent_deals" | "rent_stats";

export type EvidenceState = "grounded" | "partial_missing" | "no_evidence" | "error";

export type CharacterId = "conan" | "haibara" | "ran" | "agasa" | "ayumi";

export interface UserHousingProfile {
  household: string;
  monthlyIncome: string;
  assets: string;
  housingStatus: string;
}

export interface SavedClue {
  id: string;
  question: string;
  domain: string;
  characterName: string;
  summary: string;
  savedAt: string;
}

export interface CitationItem {
  chunk_id?: string;
  title?: string;
  source?: string;
  location?: string;
  url?: string;
}

export interface OfficialSite {
  name: string;
  url: string;
  desc: string;
}

export interface RAGResponsePayload {
  status: "answered" | "insufficient_evidence" | "unsupported" | "error";
  domain: string;
  selected_part: DomainKey;
  character_id?: CharacterId;
  resolved_query?: string;
  answer: string;
  raw_answer?: string;
  citations: CitationItem[];
  retrieved_contexts: string[];
  chunk_ids: string[];
  latency_ms: number;
  collection_name: string;
  route_reason: string;
  error_message?: string;
  meta?: {
    evidence_state?: EvidenceState;
    cache_hit?: boolean;
    reused_context?: boolean;
    conversational_mode?: boolean;
    [key: string]: unknown;
  };
  follow_ups?: string[];
  official_sites?: OfficialSite[];
}

export interface ChatTurn {
  id: string;
  question: string;
  selectedPart: DomainKey;
  response?: RAGResponsePayload;
  isLoading?: boolean;
  isNew?: boolean;
}

export interface HealthPayload {
  ok: boolean;
  registered_domains: string[];
  total_domains: number;
  error: string;
}
