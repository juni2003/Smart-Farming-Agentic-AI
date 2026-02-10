import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        leaf: {
          50: "#f3fbf5",
          100: "#e2f7e9",
          200: "#c0efd1",
          300: "#8ee0b0",
          400: "#56ca85",
          500: "#32b965",
          600: "#239550",
          700: "#1d7641",
          800: "#195f36",
          900: "#154d2e"
        },
        sun: {
          50: "#fffbe6",
          100: "#fff2b8",
          200: "#ffe48a",
          300: "#ffd45c",
          400: "#ffc22e",
          500: "#f2a900",
          600: "#c88500",
          700: "#9f6500",
          800: "#764700",
          900: "#4d2a00"
        }
      },
      backgroundImage: {
        "hero-gradient": "linear-gradient(135deg, #e2f7e9 0%, #fff2b8 100%)"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(22, 70, 44, 0.15)"
      }
    }
  },
  plugins: []
};

export default config;
