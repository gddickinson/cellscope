// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py. Everything the
// pages compute (tracking, shape/state metrics, the planner) runs in the browser, so no call needs the server.
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path) {
  return file(path.split('?')[0].replace(/^\/+/, ''));
}
