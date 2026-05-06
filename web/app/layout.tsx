import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neural Style Transfer Toolkit",
  description:
    "Hub for Gatys VGG-19, AdaIN fast NST, and temporal video stylization — run locally or via Docker.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
