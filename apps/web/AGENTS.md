# Web App — Agent Guidance

Ownership: runtime_core
Test lanes: `bash ops/run-lane.sh web-smoke`, `bash ops/run-lane.sh web-e2e`

## Stack

- React 19 + TypeScript 6 + Vite 8
- Three.js / React-Three-Fiber for 3D mesh visualization
- Vitest for unit tests
- Playwright for E2E tests (see `e2e/`)

## Development

```bash
cd apps/web
npm install
npm run dev          # dev server on :5173
npm run test         # vitest unit tests
npm run build        # production build to dist/
npx playwright test  # E2E tests (requires dev server or preview)
```

## Rules for Agents

- All API calls use `fetch()` — no direct database access from the web layer
- State management uses React hooks only — no external store libraries
- Type suppression comments (`@ts-ignore`, `eslint-disable`) are forbidden
- Use `data-testid` attributes on interactive elements for Playwright selectors
- Error states must be named components, not inline conditional soup
- The `dist/` directory is generated; never commit it

## Component Conventions

- One component per file; filename matches component name
- Use `useCallback` for stable handler refs in `useEffect` deps arrays
- Loading/error/empty states get dedicated named components (not ternary chains)
