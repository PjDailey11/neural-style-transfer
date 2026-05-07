/**
 * Vercel runs `next build` from the repo root context but discovers `web/` via workspaces.
 * After building, the platform still expects `.next/` next to `vercel.json` at the root.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const src = path.join(root, "web", ".next");
const dest = path.join(root, ".next");

if (!fs.existsSync(src)) {
  console.error("sync-web-next: missing", src);
  process.exit(1);
}

fs.rmSync(dest, { recursive: true, force: true });
fs.cpSync(src, dest, { recursive: true });
console.log("sync-web-next: copied web/.next -> .next");
