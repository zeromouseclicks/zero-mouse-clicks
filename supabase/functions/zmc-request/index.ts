// Zero Mouse Clicks — contact card on zeromouseclicks.com.
// Receives the visitor's email, a message and (optionally) the brochure image or PDF,
// stores the file in a private bucket, logs the request and forwards it to the
// AgentMail inbox with the file attached and Reply-To set to the visitor.
import { createClient } from "jsr:@supabase/supabase-js@2";

const INBOX = "zeromouseclicks@agentmail.to";
const BUCKET = "zmc-requests";
const MAX_BYTES = 15 * 1024 * 1024;
const PER_IP_PER_HOUR = 5;
const PER_DAY = 100;
const ORIGINS = [
  "https://zeromouseclicks.com",
  "http://zeromouseclicks.com",
  "https://www.zeromouseclicks.com",
  "http://www.zeromouseclicks.com",
];
const TOPICS: Record<string, string> = {
  help: "It did not work",
  more: "Need more (CAD, sections, quote)",
  feedback: "Feedback",
  other: "Message",
};

const db = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false } },
);

function cors(origin: string | null): Record<string, string> {
  const ok = origin && ORIGINS.includes(origin);
  return {
    "Access-Control-Allow-Origin": ok ? origin! : ORIGINS[0],
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Vary": "Origin",
  };
}

function reply(status: number, body: unknown, origin: string | null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors(origin), "Content-Type": "application/json" },
  });
}

// File type from the first bytes, not from what the browser claims.
function sniff(b: Uint8Array): { type: string; ext: string } | null {
  const s = (i: number, n: number) => String.fromCharCode(...b.slice(i, i + n));
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return { type: "image/jpeg", ext: "jpg" };
  if (b[0] === 0x89 && s(1, 3) === "PNG") return { type: "image/png", ext: "png" };
  if (s(0, 4) === "RIFF" && s(8, 4) === "WEBP") return { type: "image/webp", ext: "webp" };
  if (s(0, 5) === "%PDF-") return { type: "application/pdf", ext: "pdf" };
  if (s(4, 4) === "ftyp" && ["heic", "heix", "mif1", "msf1", "heif"].includes(s(8, 4))) {
    return { type: "image/heic", ext: "heic" };
  }
  return null;
}

async function sha(text: string) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(d)).map((x) => x.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

const EMAIL = /^[^\s@<>,;"]{1,64}@[^\s@<>,;"]{1,190}\.[A-Za-z]{2,24}$/;

Deno.serve(async (req) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return reply(405, { ok: false, error: "method" }, origin);
  if (!origin || !ORIGINS.includes(origin)) return reply(403, { ok: false, error: "origin" }, origin);

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return reply(400, { ok: false, error: "form" }, origin);
  }

  // Honeypot: real people never fill this field.
  if (String(form.get("website") ?? "").trim() !== "") return reply(200, { ok: true }, origin);

  const email = String(form.get("email") ?? "").trim().toLowerCase();
  const area = String(form.get("area") ?? "").trim().slice(0, 40);
  const note = String(form.get("note") ?? "").trim().slice(0, 3000);
  const detail = String(form.get("detail") ?? "").trim().slice(0, 1200);
  const topicKey = String(form.get("topic") ?? "other");
  const topic = TOPICS[topicKey] ?? TOPICS.other;
  const raw = form.get("file");
  const file = raw instanceof File && raw.size > 0 ? raw : null;
  if (!EMAIL.test(email)) return reply(400, { ok: false, error: "email" }, origin);
  if (!file && note.length < 2) return reply(400, { ok: false, error: "empty" }, origin);
  if (file && file.size > MAX_BYTES) return reply(413, { ok: false, error: "size" }, origin);

  let bytes: Uint8Array | null = null, kind: { type: string; ext: string } | null = null;
  if (file) {
    bytes = new Uint8Array(await file.arrayBuffer());
    kind = sniff(bytes);
    if (!kind) return reply(415, { ok: false, error: "type" }, origin);
  }

  // Rate limits: per visitor and overall.
  const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim();
  const ipHash = await sha(ip + "|zmc");
  const hourAgo = new Date(Date.now() - 3600e3).toISOString();
  const dayAgo = new Date(Date.now() - 86400e3).toISOString();
  const [{ count: mine }, { count: all }] = await Promise.all([
    db.from("zmc_requests").select("id", { count: "exact", head: true }).eq("ip_hash", ipHash).gte("created_at", hourAgo),
    db.from("zmc_requests").select("id", { count: "exact", head: true }).gte("created_at", dayAgo),
  ]);
  if ((mine ?? 0) >= PER_IP_PER_HOUR || (all ?? 0) >= PER_DAY) {
    return reply(429, { ok: false, error: "busy" }, origin);
  }

  const id = crypto.randomUUID();
  const safeName = file ? (file.name || "brochure").replace(/[^\w.\- ]+/g, "_").slice(0, 80) : null;
  const path = file ? `${new Date().toISOString().slice(0, 10)}/${id}.${kind!.ext}` : null;

  if (file) {
    const up = await db.storage.from(BUCKET).upload(path!, bytes!, { contentType: kind!.type, upsert: false });
    if (up.error) return reply(500, { ok: false, error: "store" }, origin);
  }

  const fullNote = [`[${topic}]`, note, detail ? `\n— ${detail}` : ""].filter(Boolean).join(" ").slice(0, 4000);
  const ins = await db.from("zmc_requests").insert({
    id, email, area: area || null, note: fullNote, file_path: path,
    file_name: safeName, file_type: kind?.type ?? null, file_size: file?.size ?? null, ip_hash: ipHash,
  });
  if (ins.error) return reply(500, { ok: false, error: "db" }, origin);

  // Forward to the inbox. The attachment is a signed link AgentMail downloads once.
  let status = "mailed", mailError: string | null = null;
  try {
    const key = Deno.env.get("AGENTMAIL_API_KEY");
    if (!key) throw new Error("AGENTMAIL_API_KEY not set");
    const attachments = [];
    if (file) {
      const signed = await db.storage.from(BUCKET).createSignedUrl(path!, 3600);
      if (signed.error) throw new Error("signed url: " + signed.error.message);
      attachments.push({ url: signed.data.signedUrl, filename: `brochure-${id.slice(0, 8)}.${kind!.ext}`, content_type: kind!.type });
    }
    const text = [
      `${topic} — from zeromouseclicks.com`,
      "",
      `From: ${email}`,
      area ? `Private area: ${area}` : "",
      "",
      note || "(no message)",
      "",
      detail ? `Technical detail: ${detail}` : "",
      file ? `File: ${safeName} (${kind!.type}, ${(file.size / 1048576).toFixed(1)} MB)` : "File: (none)",
      `Request id: ${id}`,
      "",
      "Reply to this message to answer the visitor.",
    ].filter((l, i, a) => l !== "" || a[i - 1] !== "").join("\n");
    const res = await fetch(`https://api.agentmail.to/v0/inboxes/${encodeURIComponent(INBOX)}/messages/send`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        to: INBOX,
        reply_to: email,
        subject: `${topic}: ${email}`,
        text,
        labels: ["zmc-contact", `zmc-${topicKey in TOPICS ? topicKey : "other"}`],
        attachments,
      }),
    });
    if (!res.ok) throw new Error(`agentmail ${res.status}: ${(await res.text()).slice(0, 300)}`);
  } catch (e) {
    status = "mail_failed";
    mailError = String(e).slice(0, 500);
  }
  await db.from("zmc_requests").update({ status, mail_error: mailError }).eq("id", id);

  // The request is saved either way, so the visitor gets a success answer.
  return reply(200, { ok: true, id }, origin);
});
