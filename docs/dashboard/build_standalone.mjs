// Builds index.html (a standalone page that opens from disk) from evidence_board.html (the artifact fragment).
// Run from anywhere: node docs/dashboard/build_standalone.mjs
// The fragment is the single source; index.html is regenerated, never edited by hand.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "evidence_board.html"), "utf8");

if (!src.startsWith("<title>")) throw new Error("evidence_board.html must start with <title>");
if (/<!doctype|<html|<\/html|<head[ >]|<\/head|<body|<\/body/i.test(src)) throw new Error("evidence_board.html must not carry document wrappers");

// Everything before the first <style> (title, meta, font links) goes into <head>; the rest into <body>.
const cut = src.indexOf("<style>");
if (cut < 0) throw new Error("no <style> block found");
const head = src.slice(0, cut).trim();
const body = src.slice(cut).trim();

const out = [
  "<!doctype html>",
  '<html lang="en">',
  "<head>",
  '<meta charset="utf-8">',
  '<meta name="viewport" content="width=device-width, initial-scale=1">',
  head,
  "</head>",
  "<body>",
  body,
  "</body>",
  "</html>",
  ""
].join("\n");

writeFileSync(join(here, "index.html"), out, "utf8");
console.log(`index.html written: ${Buffer.byteLength(out)} bytes from ${Buffer.byteLength(src)} bytes of fragment`);
