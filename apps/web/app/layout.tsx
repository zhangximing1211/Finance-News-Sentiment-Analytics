import type { ReactNode } from "react";
import { SidebarNav } from "../components/sidebar-nav";
import "./globals.css";

export const metadata = {
  title: "Finance Sentiment Agent",
  description: "Finance news sentiment, event tagging, and review routing dashboard.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="app-frame">
          <SidebarNav />
          <div className="app-main">{children}</div>
        </div>
      </body>
    </html>
  );
}
