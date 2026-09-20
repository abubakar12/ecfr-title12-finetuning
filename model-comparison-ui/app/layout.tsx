import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RegBench — Title 12 Model Comparison",
  description: "Compare base and fine-tuned regulatory models on identical questions, metrics, and saved evaluation history.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
