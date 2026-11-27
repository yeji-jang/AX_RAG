import type { CharacterId, DomainKey, EvidenceState, OfficialSite } from "./types";

export interface CharacterProfile {
  id: CharacterId;
  name: string;
  role: string;
  avatar: string;
  thinkingAvatar?: string;
  happyAvatar?: string;
  accent: string;
  accentBg: string;
  accentBorder: string;
  catchphrase: string;
}

export const CHARACTERS: Record<CharacterId, CharacterProfile> = {
  conan: {
    id: "conan",
    name: "코난",
    role: "총괄 탐정 · 공공임대 공고 분석",
    avatar: "/characters/conan_front.png",
    thinkingAvatar: "/characters/conan_thinking.png",
    happyAvatar: "/characters/conan_happy.png",
    accent: "#1B64DA",
    accentBg: "#EEF4FF",
    accentBorder: "#C5DBFF",
    catchphrase: "진실은 언제나 하나! 공고문 표와 조건을 정확히 짚어줄게.",
  },
  haibara: {
    id: "haibara",
    name: "홍장미",
    role: "임대차 법령 분석가",
    avatar: "/characters/haibara.png",
    accent: "#7C3AED",
    accentBg: "#F5F0FF",
    accentBorder: "#DDD0FF",
    catchphrase: "법 조문과 기한은 예외가 중요해. 주택임대차보호법 원문으로 확인해 줄게.",
  },
  ran: {
    id: "ran",
    name: "유미란",
    role: "전월세 조건 탐색 파트너",
    avatar: "/characters/ran.png",
    accent: "#0D9488",
    accentBg: "#ECFDFA",
    accentBorder: "#B9F2E7",
    catchphrase: "보증금과 월세 예산에 딱 맞는 실거래 사례를 꼼꼼히 찾아드릴게요!",
  },
  agasa: {
    id: "agasa",
    name: "브라운 박사",
    role: "서울 25구 시세 통계 박사",
    avatar: "/characters/agasa.png",
    accent: "#D97706",
    accentBg: "#FFF8EB",
    accentBorder: "#FDE0A6",
    catchphrase: "허허, 최근 1년 56만 건 실거래 통계로 우리 동네 시세를 바로 비교해 보세!",
  },
  ayumi: {
    id: "ayumi",
    name: "한아름",
    role: "서울 주거 탐색 메이트",
    avatar: "/characters/ayumi.png",
    accent: "#E11D48",
    accentBg: "#FFF1F4",
    accentBorder: "#FECDD6",
    catchphrase: "어려운 주거 용어도 쉽고 친근하게 같이 알아봐요!",
  },
};

export interface SampleGroup {
  type: "grounded" | "partial_missing" | "no_evidence";
  badge: string;
  fg: string;
  bg: string;
  border: string;
  desc: string;
  hint?: string;
  items: Array<{ label: string; question: string }>;
}

export interface DomainConfig {
  key: DomainKey;
  label: string;
  fullName: string;
  owner: string;
  collection: string;
  characterId: CharacterId;
  companionId?: CharacterId;
  greeting: string;
  desc: string;
  placeholder: string;
  samples: Array<{ label: string; question: string }>;
  sampleGroups?: SampleGroup[];
}

export const DOMAIN_ORDER: DomainKey[] = [
  "auto",
  "public_housing",
  "law",
  "rent_deals",
  "rent_stats",
];

export const DOMAIN_CONFIGS: Record<DomainKey, DomainConfig> = {
  auto: {
    key: "auto",
    label: "전체",
    fullName: "전체 자동 탐색",
    owner: "4인 통합 라우터",
    collection: "4개 도메인 자동 분기",
    characterId: "conan",
    greeting:
      "안녕! 서울에서 집 찾고 있구나? 질문을 들으면 공공임대·법령·전월세·시세 중 가장 잘 아는 친구와 함께 정확히 추리해 줄게!",
    desc: "질문 의도를 분석해 공공임대 · 임대차 법령 · 전월세 조건 · 25구 시세 중 가장 적합한 담당 캐릭터로 자동 연결합니다.",
    placeholder: "서울 공공임대, 전월세 시세, 임대차 법령 무엇이든 물어보세요",
    samples: [
      {
        label: "공공임대 · 대치르엘 35㎡ 청년 임대료",
        question:
          "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
      },
      {
        label: "임대차 법령 · 월세 인상 법적 상한",
        question: "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?",
      },
      {
        label: "전월세 조건 · 신림동 보증금 500 / 월세 45 이하",
        question: "신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘",
      },
      {
        label: "25구 시세 · 신림동 원룸 월세 중앙값",
        question: "관악구 신림동 단독다가구 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?",
      },
    ],
  },
  public_housing: {
    key: "public_housing",
    label: "공공임대",
    fullName: "공공임대 · 청년안심 공고문",
    owner: "장예지 (yj)",
    collection: "yj_public_rental · chunkV2",
    characterId: "conan",
    companionId: "ayumi",
    greeting:
      "LH·SH 공공임대 공고문 90건(2,482쪽)을 꼼꼼히 분석해 뒀어! 임대조건부터 내 소득으로 신청 가능한지까지 바로 이어서 물어봐!",
    desc: "표 구조 보존 청킹(V2)과 Hybrid RRF + LLM 리랭킹으로 입주자격 · 임대조건 · 일정을 근거와 함께 찾습니다.",
    placeholder: "공고명, 단지명, 주택형 또는 내 소득·자산 조건을 입력해 보세요",
    samples: [
      {
        label: "근거 확인 · 대치르엘 35㎡ 청년 임대조건",
        question:
          "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
      },
      {
        label: "근거 확인 · 관악봉천 행복주택 17C형 임대조건",
        question:
          "서울관악봉천 H-1·2·3BL 행복주택 예비입주자 모집공고의 17C형 기본 임대조건은 어떻게 되나요?",
      },
      {
        label: "근거 부족 · LH수서1 예상 경쟁률·커트라인",
        question:
          "LH수서1 행복주택 36A형의 2026년 11월 기준 예상 당첨 경쟁률과 커트라인 점수는 몇 점인가요?",
      },
      {
        label: "근거 없음 · 부산 해운대구 국민임대 신청 기간",
        question: "부산광역시 해운대구 국민임대주택 예비입주자 모집 신청 기간은 언제인가요?",
      },
    ],
    sampleGroups: [
      {
        type: "grounded",
        badge: "유형 1 · 근거 확인 (이어묻기 가능)",
        fg: "#0B7A6E",
        bg: "#EDFAF7",
        border: "#BEECE1",
        desc: "공고문 원문에서 임대조건·자격을 확인하고, 내 소득·자산 조건으로 이어서 상담할 수 있어요.",
        hint: '답변을 받은 뒤 "내 월소득이 350만원인데 나도 신청 가능한가?"처럼 같은 공고로 이어서 질문해 보세요.',
        items: [
          {
            label: "임대조건 → 자격 이어묻기",
            question:
              "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
          },
          {
            label: "임대조건 → 소득·서류 이어묻기",
            question:
              "서울관악봉천 H-1·2·3BL 행복주택 예비입주자 모집공고의 17C형 기본 임대조건은 어떻게 되나요?",
          },
        ],
      },
      {
        type: "partial_missing",
        badge: "유형 2 · 근거 부족 문서 (공식 사이트 + 질문 유도)",
        fg: "#A85507",
        bg: "#FFF8EB",
        border: "#F6D7A3",
        desc: "공고문에 기재되지 않은 세부 결과(경쟁률·커트라인·대기기간)를 물으면 관련 공식 사이트와 대체 질문을 안내합니다.",
        items: [
          {
            label: "경쟁률 · 커트라인 미기재",
            question:
              "LH수서1 행복주택 36A형의 2026년 11월 기준 예상 당첨 경쟁률과 커트라인 점수는 몇 점인가요?",
          },
          {
            label: "실제 경쟁률 · 예비순번 대기기간",
            question:
              "2026년 1차 행복주택 공고에서 강남구 대치르엘 35㎡ 청년 유형의 실제 청약 경쟁률과 예비순번 대기 기간은 얼마나 걸리나요?",
          },
        ],
      },
      {
        type: "no_evidence",
        badge: "유형 3 · 근거 없음 (환각 방지 + 대안 추천)",
        fg: "#475467",
        bg: "#F2F4F7",
        border: "#D8E0EB",
        desc: "수집 범위 밖(타 지역·미래 정책) 질문은 지어내지 않고 사실대로 모른다고 답한 뒤 가능한 질문을 추천합니다.",
        items: [
          {
            label: "타 지역 공고 · 수집 범위 외",
            question: "부산광역시 해운대구 국민임대주택 예비입주자 모집 신청 기간은 언제인가요?",
          },
          {
            label: "미래 정책 · 문서 없음",
            question:
              "2028년 시행 예정인 서울시 공공임대주택 소득기준 개편안의 1인 가구 상한액은 얼마인가요?",
          },
        ],
      },
    ],
  },
  law: {
    key: "law",
    label: "임대차 법령",
    fullName: "주택임대차 법령 · 가이드",
    owner: "최지훈 (jh)",
    collection: "jh_1000 · 157조문",
    characterId: "haibara",
    greeting:
      "계약갱신요구권, 월세 인상 상한, 전입신고 기한까지 현행 주택임대차보호법 157개 조문 근거를 정확히 짚어드릴게요.",
    desc: "주택임대차보호법 · 주민등록법 · 표준계약서 157조문에서 중복 제거 Hybrid RRF로 조문 근거를 안내합니다.",
    placeholder: "계약갱신요구권, 전입신고, 확정일자, 월세 증액 상한을 질문해 보세요",
    samples: [
      {
        label: "차임상한 · 월세 인상 법적 상한선",
        question: "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?",
      },
      {
        label: "신고기한 · 이사 후 전입신고 법정 기한",
        question: "이사한 뒤 전입신고는 며칠 이내에 해야 하나요?",
      },
      {
        label: "계약갱신 · 계약갱신요구권 기간과 조건",
        question: "임차인이 계약갱신요구권을 행사할 수 있는 기간과 조건이 어떻게 되나요?",
      },
      {
        label: "근거 부족 · 도배·장판 법정 벽지 단가 기준",
        question: "임대차 계약 시 도배·장판 법정 벽지 단가 기준은 얼마인가요?",
      },
    ],
  },
  rent_deals: {
    key: "rent_deals",
    label: "전월세 조건",
    fullName: "전월세 개별 거래 조건 탐색",
    owner: "선병철 (bc)",
    collection: "course_v0_small · 31,772건",
    characterId: "ran",
    greeting:
      "원하는 동네와 보증금·월세 예산을 알려주시면, 실제 전월세 실거래 31,772건 중에서 딱 맞는 사례를 찾아드릴게요!",
    desc: "관악구 실거래 31,772건에서 금액 · 면적 · 유형 조건을 추출해 공통 필터와 리랭킹으로 사례를 찾습니다.",
    placeholder: "동네, 보증금/월세 상한, 면적 조건을 입력해 보세요",
    samples: [
      {
        label: "월세 조건 · 신림동 보증금 500 / 월세 45 이하",
        question: "신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘",
      },
      {
        label: "전세 조건 · 남현동 빌라 2.5억 이하 / 8평 이상",
        question: "남현동 빌라 전세, 보증금 2억 5천만원 이하, 8평 이상 거래 사례가 있나요?",
      },
      {
        label: "원룸 조건 · 봉천동 보증금 1000 / 월세 55 이하",
        question: "봉천동에서 보증금 1000만원 이하, 월세 55만원 이하 원룸 거래를 알려줘",
      },
      {
        label: "범위 외 · 부산 해운대구 아파트 매매가",
        question: "부산 해운대구 아파트 매매가는 얼마인가요?",
      },
    ],
  },
  rent_stats: {
    key: "rent_stats",
    label: "25구 시세",
    fullName: "서울 25구 전월세 시세 통계",
    owner: "최기혁 (kh)",
    collection: "housing_rent_kh_kure_v1 · 6,200블록",
    characterId: "agasa",
    greeting:
      "서울 25개 자치구 최근 1년 실거래 56만 건을 정리해 두었네! 궁금한 동네 시세나 두 지역 비교를 물어보게나!",
    desc: "서울 25개 자치구 실거래 562,624건을 집계해 KURE-v1 + 사전 필터로 지역별 시세 중앙값을 안내합니다.",
    placeholder: "자치구, 동네, 평형대별 전월세 시세나 지역 비교를 질문해 보세요",
    samples: [
      {
        label: "원룸 시세 · 관악구 신림동 33㎡ 이하 월세",
        question: "관악구 신림동 단독다가구 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?",
      },
      {
        label: "지역 비교 · 강남 삼성동 전세 vs 노원 월계동 월세",
        question:
          "강남구 삼성동 빌라 전세(33㎡ 이하)와 노원구 월계동 단독다가구 월세(33~60㎡) 시세를 비교해줘",
      },
      {
        label: "전세 시세 · 용산구 한남동 33~60㎡ 전세 중앙값",
        question: "용산구 한남동 단독다가구 전세 33~60㎡ 보증금 중앙값은 얼마인가요?",
      },
      {
        label: "근거 없음 · 2027년 강남구 원룸 월세 전망",
        question: "2027년 서울 강남구 원룸 월세 전망치는 얼마로 예상되나요?",
      },
    ],
  },
};

export const DOMAIN_LABELS: Record<string, string> = {
  public_housing: "공공임대 공고",
  law: "임대차 법령",
  rent_deals: "전월세 조건",
  rent_stats: "25구 시세",
  unsupported: "지원 범위 외",
};

export const EVIDENCE_STYLE: Record<
  EvidenceState,
  { label: string; fg: string; bg: string; dot: string }
> = {
  grounded: { label: "근거 확인", fg: "#0B7A6E", bg: "#EDFAF7", dot: "#0B7A6E" },
  partial_missing: { label: "근거 부족", fg: "#A85507", bg: "#FFF8EB", dot: "#D97706" },
  no_evidence: { label: "근거 없음", fg: "#475467", bg: "#F2F4F7", dot: "#64748B" },
  error: { label: "오류 격리", fg: "#C92A2A", bg: "#FFF2F2", dot: "#DC2626" },
};

export const FALLBACK_OFFICIAL_SITES: Record<string, OfficialSite[]> = {
  public_housing: [
    {
      name: "SH 서울주택도시공사 인터넷청약시스템",
      url: "https://www.i-sh.co.kr/app/index.do",
      desc: "행복주택·청년안심주택 공고문, 청약 경쟁률·당첨자·예비순번 조회",
    },
    {
      name: "LH 청약플러스",
      url: "https://apply.lh.or.kr",
      desc: "LH 공공임대·매입임대·예비입주자 모집공고 및 청약 결과 확인",
    },
    {
      name: "마이홈 포털 (국토교통부)",
      url: "https://www.myhome.go.kr",
      desc: "공공임대 입주자격 자가진단 및 전국 임대주택 공고 통합 조회",
    },
    {
      name: "서울주거포털",
      url: "https://housing.seoul.go.kr",
      desc: "서울시 청년안심주택·미리내집·주거지원 정책 공식 안내",
    },
  ],
};

export function getCharacterForDomain(domain?: string, selectedPart: DomainKey = "auto"): CharacterProfile {
  if (domain === "law") return CHARACTERS.haibara;
  if (domain === "rent_deals") return CHARACTERS.ran;
  if (domain === "rent_stats") return CHARACTERS.agasa;
  if (domain === "public_housing") return CHARACTERS.conan;
  return CHARACTERS[DOMAIN_CONFIGS[selectedPart]?.characterId ?? "conan"];
}
