# 서울 주거 탐색 — Next.js 15 대화형 UI

4도메인 주거 RAG(공공임대 공고 · 임대차 법령 · 전월세 조건 · 서울 25구 시세)와 연동되는 Next.js 15 프론트엔드 및 FastAPI 백엔드 브리지입니다.

> **저작권 안내**: 공개 저장소 저작권 보호를 위해 로컬 시연용 캐릭터 및 배경 이미지(`asset/`, `public/characters/`, `public/scenes/`)는 `.gitignore`로 제외되어 있습니다. 로컬 커스텀 이미지를 배치하면 자동으로 표시됩니다.

## 실행 방법 (2개 터미널)

### 1) FastAPI 백엔드 서버 실행 (`rag-home-api` 루트)
```powershell
cd c:\Users\wkddp\Documents\RAG_Team\rag-home-api
uv run uvicorn rag_home_api.nextjs_ui.api_server:app --port 8000
```

### 2) Next.js 프론트엔드 실행 (`nextjs_ui` 폴더)
> Node.js 20 LTS 이상이 필요합니다.

```powershell
cd c:\Users\wkddp\Documents\RAG_Team\rag-home-api\src\rag_home_api\nextjs_ui
npm install
npm run dev
# 프로덕션 모드: npm run build && npm run start
```

브라우저에서 **http://localhost:3000** 접속!
