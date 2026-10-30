import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import test from 'node:test';
import { decodeKey, decryptHeader, encryptHeader, validateInputSize } from '../ream_header_backup.mjs';

test('authenticated backup round-trips without exposing the plaintext', () => {
  const plain = Buffer.from('tiny synthetic metadata; no model weights'.repeat(50));
  const key = randomBytes(32), sha = createHash('sha256').update(plain).digest('hex');
  const first = encryptHeader(plain, key, plain.length, sha);
  const second = encryptHeader(plain, key, plain.length, sha);
  assert.notDeepEqual(first, second);
  assert.equal(first.includes(plain), false);
  assert.deepEqual(decryptHeader(first, key, plain.length, sha), plain);
  assert.throws(() => decryptHeader(first, randomBytes(32), plain.length, sha));
  assert.throws(() => decryptHeader(first, key, plain.length, '0'.repeat(64)));
  const corrupted = Buffer.from(first); corrupted[corrupted.length - 1] ^= 1;
  assert.throws(() => decryptHeader(corrupted, key, plain.length, sha));
  assert.throws(() => encryptHeader(plain, key, plain.length + 1, sha));
});

test('only canonical 32-byte keys are accepted', () => {
  const key = randomBytes(32);
  assert.deepEqual(decodeKey(key.toString('base64')), key);
  for (const value of ['', 'not-base64', Buffer.alloc(16).toString('base64'), undefined])
    assert.throws(() => decodeKey(value));
});

test('CLI input size bounds reject full model files before reading', () => {
  assert.doesNotThrow(() => validateInputSize('encrypt', 11_023_104));
  assert.doesNotThrow(() => validateInputSize('restore', 3_130_868));
  for (const [mode, size] of [['encrypt', 56_115_609_536], ['encrypt', 0],
                               ['restore', 56_115_609_536], ['restore', 10], ['other', 100]])
    assert.throws(() => validateInputSize(mode, size));
});
