# 구해줘 홈즈 — Next.js 캐릭터 대화형 UI

서울 야경 인트로(코난 뒷모습 → 마우스 오버 시 뒤돌아 첫 질문 안내)와 4개 도메인 캐릭터 대화방(공공임대 · 임대차 법령 · 전월세 조건 · 25구 시세)을 제공하는 Next.js 15 프론트엔드 및 FastAPI 백엔드 브리지입니다.

## 실행 방법 (2개 터미널)

### 1) FastAPI 백엔드 서버 실행 (`rag-home-api` 루트)
```powershell
cd c:\Users\wkddp\Documents\RAG_Team\rag-home-api
uv run uvicorn rag_home_api.nextjs_ui.api_server:app --reload --port 8000
```

### 2) Next.js 프론트엔드 실행 (`nextjs_ui` 폴더)
> Node.js 20 LTS 이상이 필요합니다. 미설치 시 `winget install OpenJS.NodeJS.LTS` 후 새 터미널을 열어 주세요.

```powershell
cd c:\Users\wkddp\Documents\RAG_Team\rag-home-api\src\rag_home_api\nextjs_ui
npm install
npm run dev
```

브라우저에서 **http://localhost:3000** 접속!
