import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const html = readFileSync(new URL('../_static/presenter.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('const sourceUrl='), html.indexOf('const doc='));

function sourceUrl(href, { page, referrer = '' } = {}) {
  return runInNewContext(`${source}\nsourceUrl()`, {
    URL, URLSearchParams,
    location: new URL(href),
    document: { referrer },
    window: { frameElement: page ? { dataset: { page } } : null },
  });
}

test('embedded presenter uses the complete chapter URL, including its prefix and anchor', () => {
  const page = 'http://localhost:3000/rlbook/stochastic-dp#uncertainty';
  assert.equal(sourceUrl('about:srcdoc', { page }), page);
});

test('standalone presenter resolves explicit pages and rejects cross-origin pages', () => {
  const presenter = 'https://example.org/rlbook/build/presenter-abc.html';
  assert.equal(sourceUrl(`${presenter}?page=../stochastic-dp/`), 'https://example.org/rlbook/stochastic-dp/');
  assert.equal(sourceUrl(`${presenter}?page=https://other.example/chapter`), 'https://example.org/rlbook/modeling-controlled-systems/');
});

test('fallback retains the book prefix for built assets and source previews', () => {
  for (const asset of ['build/presenter-abc.html', '_static/presenter.html', 'presenter-abc.html']) {
    assert.equal(sourceUrl(`https://example.org/rlbook/${asset}`), 'https://example.org/rlbook/modeling-controlled-systems/');
  }
});
