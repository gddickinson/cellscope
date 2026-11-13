// Write a NumPy .npz (ZIP of .npy files, deflate via the platform's CompressionStream) so a label stack tracked in
// the browser can be opened by CellScope (key "labels", like masks.npz) or CellScope Analysis. No dependencies.

const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
function crc32(u8) { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }

async function deflateRaw(u8) {
  const s = new Blob([u8]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

// .npy bytes for an Int32Array of the given shape (little-endian '<i4', C order)
export function npyBytes(data, shape) {
  let h = `{'descr': '<i4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = 10 + h.length + 1, padN = (64 - (total % 64)) % 64;
  h += ' '.repeat(padN) + '\n';
  const out = new Uint8Array(10 + h.length + data.byteLength);
  out.set([0x93, 78, 85, 77, 80, 89, 1, 0, h.length & 0xff, h.length >> 8]);
  out.set(new TextEncoder().encode(h), 10);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + h.length);
  return out;
}

// arrays: {name: {data: Int32Array, shape}} -> Blob (.npz)
export async function npzBlob(arrays) {
  const parts = [], central = [];
  let offset = 0;
  for (const [name, { data, shape }] of Object.entries(arrays)) {
    const raw = npyBytes(data, shape), comp = await deflateRaw(raw), fname = new TextEncoder().encode(name + '.npy');
    if (raw.length >= 0xffffffff || offset >= 0xffffffff) throw new Error('array too large for this writer (> 4 GB)');
    const crc = crc32(raw), loc = new DataView(new ArrayBuffer(30));
    [[0, 0x04034b50, 4], [4, 20, 2], [6, 0, 2], [8, 8, 2], [10, 0, 2], [12, 0, 2], [14, crc, 4], [18, comp.length, 4],
      [22, raw.length, 4], [26, fname.length, 2], [28, 0, 2]].forEach(([o, v, n]) => (n === 4 ? loc.setUint32(o, v, true) : loc.setUint16(o, v, true)));
    const cen = new DataView(new ArrayBuffer(46));
    [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0, 2], [10, 8, 2], [12, 0, 2], [14, 0, 2], [16, crc, 4],
      [20, comp.length, 4], [24, raw.length, 4], [28, fname.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2], [36, 0, 2],
      [38, 0, 4], [42, offset, 4]].forEach(([o, v, n]) => (n === 4 ? cen.setUint32(o, v, true) : cen.setUint16(o, v, true)));
    parts.push(new Uint8Array(loc.buffer), fname, comp);
    central.push(new Uint8Array(cen.buffer), fname);
    offset += 30 + fname.length + comp.length;
  }
  const cdSize = central.reduce((a, b) => a + b.length, 0), n = Object.keys(arrays).length, end = new DataView(new ArrayBuffer(22));
  [[0, 0x06054b50, 4], [8, n, 2], [10, n, 2], [12, cdSize, 4], [16, offset, 4]].forEach(([o, v, k]) => (k === 4 ? end.setUint32(o, v, true) : end.setUint16(o, v, true)));
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: 'application/octet-stream' });
}
