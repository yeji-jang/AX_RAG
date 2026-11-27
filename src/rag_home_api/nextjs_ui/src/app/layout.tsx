import type { Metadata } from "next";
import { Gowun_Dodum } from "next/font/google";
import "./globals.css";

const gowunDodum = Gowun_Dodum({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-gowun",
  display: "swap",
});

export const metadata: Metadata = {
  title: "구해줘 홈즈 — 서울 주거 탐색",
  description:
    "서울 공공임대 공고문 · 주택임대차 법령 · 전월세 실거래 · 25구 시세를 코난과 친구들과 대화하며 찾는 서울 주거 탐색 서비스",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className={gowunDodum.variable}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
