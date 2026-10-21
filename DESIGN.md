---
version: 1.0.0
name: Apple macOS Liquid Glass RAG Studio
description: Apple Human Interface Guidelines (HIG) and Liquid Glass design tokens for the Production RAG Studio interface.
colors:
  background: "#0d0f12"
  background-secondary: "#16191f"
  surface: "rgba(28, 32, 42, 0.72)"
  surface-hover: "rgba(38, 44, 58, 0.85)"
  surface-active: "rgba(48, 56, 74, 0.95)"
  glass-border: "rgba(255, 255, 255, 0.08)"
  glass-border-focus: "rgba(0, 122, 255, 0.55)"
  glass-highlight: "rgba(255, 255, 255, 0.12)"
  text-primary: "#f5f5f7"
  text-secondary: "#86868b"
  text-tertiary: "#515154"
  accent-blue: "#0071e3"
  accent-blue-hover: "#0077ed"
  accent-green: "#34c759"
  accent-orange: "#ff9f0a"
  accent-purple: "#af52de"
  accent-red: "#ff453a"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro', 'Helvetica Neue', sans-serif"
    fontSize: 28px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: -0.025em
  title-1:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro', 'Helvetica Neue', sans-serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.015em
  title-2:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro', 'Helvetica Neue', sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: -0.01em
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro', 'Helvetica Neue', sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: -0.005em
  caption:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro', 'Helvetica Neue', sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: 0.01em
  code:
    fontFamily: "'SF Mono', Menlo, Monaco, Consolas, monospace"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: 0em
rounded:
  xs: 4px
  sm: 6px
  md: 10px
  lg: 16px
  xl: 22px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
components:
  window-panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: 20px
  button-primary:
    backgroundColor: "{colors.accent-blue}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  citation-badge:
    backgroundColor: "rgba(0, 113, 227, 0.15)"
    textColor: "{colors.accent-blue}"
    rounded: "{rounded.full}"
    padding: "2px 8px"
---

# Apple macOS Liquid Glass RAG Studio — Design Specification

## Overview
This design specification defines the visual language, elevation, materials, and component standards for the **Production RAG Engine Studio**. Built strictly against Apple's **Human Interface Guidelines (HIG)** and the **Liquid Glass** material spec (`apple-design` skill), the interface brings a native macOS Tahoe/Sonoma windowing aesthetic to the AI engineering workbench:
- Subsurface scattering and frosted acrylic blur (`backdrop-filter: blur(24px) saturate(180%)`)
- Monochromatic, high-legibility San Francisco typography
- Three-column macOS split-view layout: Sidebar navigation & knowledge bases, conversational query & answer canvas, and a slide-out Chunk & Citation Inspector
- Real-time quantitative eval gauges (Faithfulness, Recall@k, Citation Precision, Latency)

## Colors
- **System Backgrounds**: Deep cosmic gray (`#0d0f12`) with secondary layered chrome (`#16191f`).
- **Liquid Glass Surface**: `rgba(28, 32, 42, 0.72)` with a 1px specular inner rim (`rgba(255, 255, 255, 0.08)`).
- **Apple Accents**:
  - San Francisco Blue (`#0071e3`): Primary action buttons, active navigation states, grounded citation links.
  - Emerald Green (`#34c759`): High confidence (>90%) faithfulness verification and optimal latency badges (<250ms).
  - Amber Orange (`#ff9f0a`): Medium confidence / warning flags.
  - Crimson Red (`#ff453a`): Hallucination alerts / unverified quotes.

## Typography
Using Apple's San Francisco system font stack with optical sizing:
- **SF Pro Display** for window headers and scorecards (`letterSpacing: -0.02em`).
- **SF Pro Text** for body prose, chat interaction, and metadata.
- **SF Mono** for JSON schemas, vector chunk hashes, token counts, and similarity scores.

## Layout
A macOS **NavigationSplitView** architecture:
1. **Sidebar (260px)**: Knowledge base management, document upload dropzone, hybrid search parameter controls (BM25 vs Dense alpha slider, reranker toggle), and saved benchmark runs.
2. **Main Canvas (Flex 1)**: Grounded query interaction, streaming response display, inline citation footnotes (`[1]`, `[2]`), and eval metrics ribbon.
3. **Inspector Panel (340px)**: Inspects retrieved chunks, compares dense similarity vs BM25 score, displays RRF rank fusion mathematics, and validates exact quote provenance.

## Elevation & Depth
- **Level 0 (Canvas)**: Deep cosmic canvas background.
- **Level 1 (Panels)**: Frosted acrylic translucent window panels with `backdrop-filter: blur(24px) saturate(180%)`.
- **Level 2 (Popovers & Modals)**: Elevated glass with specular highlight (`box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15)`).

## Shapes
- **Panels**: `16px` border-radius with hairline 1px borders.
- **Buttons & Inputs**: `8px` to `10px` border-radius following Apple macOS control metrics.
- **Citation Badges & Status Pills**: Full pill geometry (`9999px`).

## Components
- **Studio Header Bar**: Clean web application header with brand vector logo, title, connection status indicator, and segmented view switcher.
- **Segmented Control**: Pill-shaped segmented switcher with sliding highlight indicator (`All`, `RAG Search`, `Eval Scorecard`, `API Docs`).
- **Citation Footnote**: Clickable Apple-blue pill (`[1]`) that highlights and smoothly auto-scrolls the corresponding source snippet in the Inspector panel.
- **Metric Meter**: Compact Apple Health/Activity inspired radial or linear progress meter for Faithfulness, Recall, and Latency.

## Do's and Don'ts
- **DO** use subtle specular top highlights (`inset 0 1px 0 rgba(255, 255, 255, 0.12)`) on glass cards.
- **DO** maintain at least 4.5:1 text contrast for all body copy (WCAG AA/AAA).
- **DO** provide clear visual feedback when clicking citations to reveal source provenance.
- **DON'T** use heavy artificial neon gradients or generic unstyled cards.
- **DON'T** overwhelm the screen with non-functional decoration; every visual element must represent active system data.
