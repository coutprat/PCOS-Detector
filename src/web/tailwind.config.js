/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        "medical-blue": "#1e3a5f",
        "medical-teal": "#0ea5e9",
        "medical-navy": "#0f172a",
      },
      backgroundImage: {
        "gradient-medical":
          "linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%)",
      },
    },
  },
  plugins: [],
};
