import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Bananenblick – Reife auf einen Blick",
  description:
    "Erkenne die sichtbare Bananenreife direkt im Live-Kamerabild. Privat, verständlich und lokal auf deinem Handy.",
  icons: { icon: "/favicon.svg" },
};
export const viewport = { width: "device-width", initialScale: 1, themeColor: "#f8f9f4" };
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}
