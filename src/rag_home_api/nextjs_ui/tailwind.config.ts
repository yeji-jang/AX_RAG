import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Pretendard Variable",
          "Pretendard",
          "-apple-system",
          "BlinkMacSystemFont",
          "Apple SD Gothic Neo",
          "Malgun Gothic",
          "sans-serif",
        ],
      },
      colors: {
        ink: "#191F28",
        body: "#333D4B",
        muted: "#6B7684",
        subtle: "#8B95A1",
        accent: "#1B64DA",
        "accent-soft": "#EEF4FF",
        surface: "#F4F6F9",
      },
      boxShadow: {
        editorial: "0 10px 30px rgba(15, 23, 42, 0.08), 0 2px 8px rgba(15, 23, 42, 0.04)",
        bubble: "0 12px 28px rgba(15, 23, 42, 0.07), 0 1px 3px rgba(15, 23, 42, 0.05)",
        glow: "0 0 40px rgba(59, 130, 246, 0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
