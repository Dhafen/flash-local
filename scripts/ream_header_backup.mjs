// Encrypted metadata-only backup for a pinned, formerly public GGUF header.
// The authenticated plaintext is limited to the original header: no weights.
import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync, gunzipSync } from 'node:zlib';

const MAGIC = Buffer.from('FLREAM1\0', 'ascii');
const HEADER_BYTES = 11_023_104;
const HEADER_SHA256 = '352b87fd036c3b14c4bcb3ff83c19dc1802be26c93686b4300972e9a43390292';
const MAX_CIPHERTEXT = 8 * 1024 * 1024;
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const aad = (size, hash) => Buffer.from(`flash-local-ream-header-v1:${size}:${hash}`, 'ascii');

export function decodeKey(value) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9+/]{43}=$/.test(value)) throw new Error('Invalid backup key');
  const key = Buffer.from(value, 'base64');
  if (key.length !== 32 || key.toString('base64') !== value) throw new Error('Invalid backup key');
  return key;
}

export function validateInputSize(mode, size) {
  if (!Number.isSafeInteger(size) || size < 0 ||
      (mode === 'encrypt' && size !== HEADER_BYTES) ||
      (mode === 'restore' && (size <= MAGIC.length + 28 || size > MAX_CIPHERTEXT)) ||
      !['encrypt', 'restore'].includes(mode))
    throw new Error('Input size is outside the metadata-only bounds');
}

export function encryptHeader(plain, key, expectedSize, expectedHash, iv = randomBytes(12)) {
  if (key.length !== 32 || iv.length !== 12 || plain.length !== expectedSize || digest(plain) !== expectedHash)
    throw new Error('Header or encryption parameters differ from the pinned identity');
  const cipher = createCipheriv('aes-256-gcm', key, iv);
  cipher.setAAD(aad(expectedSize, expectedHash));
  const body = Buffer.concat([cipher.update(gzipSync(plain, { level: 6 })), cipher.final()]);
  const result = Buffer.concat([MAGIC, iv, cipher.getAuthTag(), body]);
  if (result.length > MAX_CIPHERTEXT) throw new Error('Encrypted header exceeds the size bound');
  return result;
}

export function decryptHeader(data, key, expectedSize, expectedHash) {
  if (key.length !== 32 || data.length <= MAGIC.length + 28 || data.length > MAX_CIPHERTEXT ||
      !data.subarray(0, MAGIC.length).equals(MAGIC)) throw new Error('Invalid encrypted header');
  const iv = data.subarray(MAGIC.length, MAGIC.length + 12);
  const tag = data.subarray(MAGIC.length + 12, MAGIC.length + 28);
  const decipher = createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAAD(aad(expectedSize, expectedHash)); decipher.setAuthTag(tag);
  const compressed = Buffer.concat([decipher.update(data.subarray(MAGIC.length + 28)), decipher.final()]);
  const plain = gunzipSync(compressed, { maxOutputLength: expectedSize });
  if (plain.length !== expectedSize || digest(plain) !== expectedHash) throw new Error('Restored header hash differs');
  return plain;
}

function main() {
  const [mode, input, output] = process.argv.slice(2);
  if (!['encrypt', 'restore'].includes(mode) || !input || !output) throw new Error('Expected mode, input and output');
  const key = decodeKey(process.env.FLASH_LOCAL_REAM_HEADER_KEY);
  const inputStat = statSync(input);
  if (!inputStat.isFile()) throw new Error('Expected a regular input file');
  validateInputSize(mode, inputStat.size);
  const bytes = readFileSync(input);
  const result = mode === 'encrypt' ? encryptHeader(bytes, key, HEADER_BYTES, HEADER_SHA256)
                                    : decryptHeader(bytes, key, HEADER_BYTES, HEADER_SHA256);
  mkdirSync(dirname(output), { recursive: true });
  if (existsSync(output)) throw new Error('Refusing to overwrite an existing file');
  const temporary = `${output}.tmp-${randomBytes(8).toString('hex')}`;
  let written = false;
  try {
    writeFileSync(temporary, result, { flag: 'wx', mode: 0o600 });
    written = true;
    renameSync(temporary, output);
    written = false;
  } finally { if (written) rmSync(temporary, { force: true }); }
  console.log(mode === 'restore' ? 'Pinned REAM header restored and SHA-256 verified; no model weights.'
                                 : 'Pinned metadata-only header encrypted successfully.');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  try { main(); } catch (error) { console.error(`REAM header backup failed: ${error.message}`); process.exitCode = 1; }
}
