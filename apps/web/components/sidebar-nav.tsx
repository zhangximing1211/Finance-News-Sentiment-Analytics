"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "总览", short: "概览" },
  { href: "/analyze", label: "单条分析", short: "分析" },
  { href: "/batch", label: "批量分析", short: "批量" },
  { href: "/watchlist", label: "Watchlist", short: "盯盘" },
  { href: "/trends", label: "情绪趋势", short: "趋势" },
  { href: "/errors", label: "错误案例", short: "错误" },
  { href: "/feedback", label: "人工反馈", short: "反馈" },
];

export function SidebarNav() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <span className="brand-kicker">Finance News Sentiment Analytics</span>
        <strong className="brand-title">Agent Console</strong>
        <p className="brand-copy">分析、复判、反馈、重训放在一条产品链路里。</p>
      </div>

      <nav className="nav-grid" aria-label="Primary">
        {navItems.map((item) => {
          const active = pathname === item.href;
          return (
            <Link className={`nav-link${active ? " nav-link-active" : ""}`} href={item.href} key={item.href}>
              <span className="nav-short">{item.short}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
