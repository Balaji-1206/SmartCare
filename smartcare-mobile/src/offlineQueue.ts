import NetInfo from "@react-native-community/netinfo";
import AsyncStorage from "@react-native-async-storage/async-storage";

const QUEUE_KEY = "smartcare_offline_queue"; // array of {url, body}

type QueueItem = { url: string; body: any };

async function readQueue(): Promise<QueueItem[]> {
  try { const raw = await AsyncStorage.getItem(QUEUE_KEY); return raw ? JSON.parse(raw) : []; }
  catch { return []; }
}
async function writeQueue(items: QueueItem[]) {
  try { await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(items)); } catch {}
}

export async function enqueue(url: string, body: any) {
  const q = await readQueue();
  q.push({ url, body });
  await writeQueue(q);
}

export async function flush(apiBase: string) {
  const state = await NetInfo.fetch();
  if (!state.isConnected) return { ok: false, count: 0 };

  let q = await readQueue();
  let success = 0;
  const keep: QueueItem[] = [];

  for (const item of q) {
    try {
      const r = await fetch(`${apiBase}${item.url}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(item.body),
      });
      if (r.ok) success += 1;
      else keep.push(item);
    } catch {
      keep.push(item);
    }
  }
  await writeQueue(keep);
  return { ok: true, count: success, remaining: keep.length };
}

export async function onlineOrQueue(apiBase: string, url: string, body: any) {
  const state = await NetInfo.fetch();
  if (!state.isConnected) {
    await enqueue(url, body);
    return { queued: true };
  }
  try {
    const r = await fetch(`${apiBase}${url}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(String(r.status));
    return { queued: false, data: await r.json() };
  } catch {
    await enqueue(url, body);
    return { queued: true };
  }
}
