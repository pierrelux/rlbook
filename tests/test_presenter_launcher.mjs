import assert from 'node:assert/strict';
import test from 'node:test';

import launcher from '../_static/presenter-launcher.mjs';

// Only the browser surface used by the launcher is mocked. The module itself
// handles the click, fetches the presenter, and creates/removes the overlay.
function browserFixture(page) {
  const listeners = new Map();
  const requests = [];
  const errors = [];
  const timers = new Map();
  let timerId = 0;
  let response = { ok: true, text: async () => '<!doctype html><html><body>Presenter</body></html>' };
  const document = {
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    },
    removeEventListener(type, listener) {
      listeners.get(type)?.delete(listener);
    },
    getElementById(id) {
      return elements().find((element) => element.id === id) ?? null;
    },
    querySelector(selector) {
      if (selector.startsWith('#')) return this.getElementById(selector.slice(1));
      return null;
    },
    createElement(tagName) {
      return element(tagName);
    },
  };
  function element(tagName) {
    const attributes = new Map();
    return {
      tagName: tagName.toUpperCase(),
      ownerDocument: document,
      children: [],
      style: { overflow: '' },
      dataset: {},
      setAttribute(name, value) {
        attributes.set(name, String(value));
        if (name === 'id') this.id = String(value);
      },
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      removeAttribute(name) {
        attributes.delete(name);
      },
      append(...children) {
        for (const child of children) {
          child.parentNode = this;
          this.children.push(child);
        }
      },
      appendChild(child) {
        this.append(child);
        return child;
      },
      replaceChildren(...children) {
        for (const child of [...this.children]) child.remove();
        this.append(...children);
      },
      remove() {
        if (this.parentNode) {
          this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
          this.parentNode = null;
        }
      },
      focus() {
        document.activeElement = this;
      },
      get isConnected() {
        return Boolean(this.parentNode);
      },
    };
  }
  function elements() {
    const descendants = (node) => [node, ...node.children.flatMap(descendants)];
    return descendants(document.documentElement);
  }
  document.documentElement = element('html');
  document.body = element('body');
  document.documentElement.append(document.body);
  document.activeElement = document.body;
  const window = {
    document,
    location: new URL(page),
    scrollX: 0,
    scrollY: 240,
    scrollTo(x, y) {
      this.scrollX = x;
      this.scrollY = y;
    },
    fetch: async (url, options) => {
      requests.push({ url: String(url), options });
      return response;
    },
    alert: (message) => errors.push(message),
    console: { error: (...args) => errors.push(args.join(' ')) },
    requestAnimationFrame: (callback) => callback(),
    setTimeout(callback) {
      timers.set(++timerId, callback);
      return timerId;
    },
    clearTimeout(id) { timers.delete(id); },
  };
  window.parent = window;
  window.top = window;
  document.defaultView = window;
  const el = element('div');
  document.body.append(el);

  return {
    window,
    document,
    el,
    requests,
    errors,
    listeners,
    flushTimers() {
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach((callback) => callback());
    },
    setResponse(value) { response = value; },
    overlay: () => document.getElementById('rl-presenter-overlay'),
    async click(href) {
      const anchor = element('a');
      anchor.href = new URL(href, window.location.href).href;
      anchor.setAttribute('href', href);
      anchor.closest = (selector) => selector === 'a[href]' ? anchor : null;
      document.body.append(anchor);
      anchor.focus();
      const event = {
        target: anchor,
        button: 0,
        defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; },
        stopPropagation() {},
        stopImmediatePropagation() {},
      };
      for (const listener of listeners.get('click') ?? []) await listener(event);
      // Browsers do not await event listeners; flush any detached async work.
      await new Promise(setImmediate);
      return { event, anchor };
    },
  };
}

function withBrowser(page, callback) {
  return callback(browserFixture(page));
}

test('split-port preview fetches the asset and presents the current chapter', async () => {
  await withBrowser('http://localhost:3000/model-predictive-path-integral-control#trajectory-weights', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    const asset = 'http://localhost:3100/presenter-a12b34c56d.html';
    const { event } = await browser.click(asset);
    assert.equal(event.defaultPrevented, true);
    assert.equal(browser.requests[0].url, asset);
    const overlay = browser.overlay();
    assert.ok(overlay);
    assert.equal(overlay.dataset.page, browser.window.location.href);
    assert.match(overlay.srcdoc, /Presenter/);
    assert.equal(overlay.src, undefined, 'the presenter must inherit the chapter origin through srcdoc');
    cleanup?.();
  });
});

test('production prefix is preserved and the chapter is captured at click time', async () => {
  await withBrowser('https://example.org/rlbook/modeling-controlled-systems/', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    browser.window.location = new URL('https://example.org/rlbook/stochastic-dp/#finite-horizon');
    const asset = 'https://example.org/rlbook/presenter-0123456789abcdef.html';
    await browser.click(asset);
    assert.equal(browser.requests[0].url, asset);
    assert.equal(browser.overlay().dataset.page, browser.window.location.href);
    cleanup?.();
  });
});

test('multiple widget mounts install one click handler and clean it up', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanupFirst = launcher.render({ el: browser.el });
    const second = browser.document.createElement('div');
    const cleanupSecond = launcher.render({ el: second });
    assert.equal(browser.listeners.get('click')?.size, 1);
    await browser.click('http://localhost:3100/presenter.html');
    assert.equal(browser.requests.length, 1);
    cleanupFirst?.();
    browser.flushTimers();
    assert.equal(browser.listeners.get('click')?.size, 1);
    assert.ok(browser.overlay());
    cleanupSecond?.();
    browser.flushTimers();
    assert.equal(browser.listeners.get('click')?.size, 0);
    assert.equal(browser.overlay(), null);
  });
});

test('ordinary chapter and external links are untouched', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    for (const url of ['/stochastic-dp', 'https://github.com/pierrelux/rlbook', '/not-presenter.html']) {
      const { event } = await browser.click(url);
      assert.equal(event.defaultPrevented, false);
    }
    assert.equal(browser.requests.length, 0);
    assert.equal(browser.overlay(), null);
    cleanup?.();
  });
});

test('exiting restores chapter scrolling and keyboard focus', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    browser.document.documentElement.style.overflow = 'auto';
    const cleanup = launcher.render({ el: browser.el });
    const { anchor } = await browser.click('http://localhost:3100/presenter.html');
    const overlay = browser.overlay();
    assert.equal(browser.document.documentElement.style.overflow, 'hidden');
    assert.equal(typeof overlay.rlPresenterExit, 'function');
    overlay.rlPresenterExit();
    assert.equal(browser.overlay(), null);
    assert.equal(browser.document.documentElement.style.overflow, 'auto');
    assert.equal(browser.document.activeElement, anchor);
    cleanup?.();
  });
});

test('a failed download leaves the chapter usable and can be retried', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    browser.setResponse({ ok: false, status: 503, text: async () => 'Unavailable' });
    await browser.click('http://localhost:3100/presenter.html');
    assert.equal(browser.overlay(), null);
    assert.equal(browser.document.documentElement.style.overflow, '');
    assert.equal(browser.el.children[0]?.getAttribute('role'), 'alert');
    browser.setResponse({ ok: true, text: async () => '<html><body>Presenter</body></html>' });
    await browser.click('http://localhost:3100/presenter.html');
    assert.equal(browser.requests.length, 2);
    assert.ok(browser.overlay());
    assert.equal(browser.el.children.length, 0, 'retry clears the previous error');
    cleanup?.();
  });
});

test('changing chapters during the download cancels that launch', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    let finish;
    const download = new Promise((resolve) => { finish = resolve; });
    browser.setResponse({ ok: true, text: () => download });
    const pendingClick = browser.click('http://localhost:3100/presenter.html');
    browser.window.location = new URL('http://localhost:3000/stochastic-dp');
    finish('<html><body>Presenter</body></html>');
    await pendingClick;
    assert.equal(browser.overlay(), null);
    assert.equal(browser.document.documentElement.style.overflow, '');
    await browser.click('http://localhost:3100/presenter.html');
    assert.equal(browser.overlay().dataset.page, browser.window.location.href);
    cleanup?.();
  });
});

test('unmounting during the download prevents a late overlay', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanup = launcher.render({ el: browser.el });
    let finish;
    const download = new Promise((resolve) => { finish = resolve; });
    browser.setResponse({ ok: true, text: () => download });
    const pendingClick = browser.click('http://localhost:3100/presenter.html');
    cleanup();
    browser.flushTimers();
    finish('<html><body>Presenter</body></html>');
    await pendingClick;
    assert.equal(browser.overlay(), null);
    assert.equal(browser.listeners.get('click')?.size, 0);
    assert.equal(browser.document.documentElement.style.overflow, '');
  });
});

test('the nested chapter inside a presenter does not install another launcher', () => {
  const browser = browserFixture('http://localhost:3000/modeling-controlled-systems');
  browser.window.top = {};
  launcher.render({ el: browser.el });
  assert.equal(browser.listeners.get('click')?.size ?? 0, 0);
});

test('widget remounting preserves the active presenter until actual removal', async () => {
  await withBrowser('http://localhost:3000/model-predictive-path-integral-control', async (browser) => {
    const cleanupFirst = launcher.render({ el: browser.el });
    const { anchor } = await browser.click('http://localhost:3100/presenter.html');
    const overlay = browser.overlay();
    cleanupFirst();
    assert.equal(browser.overlay(), overlay, 'cleanup is deferred while MyST replaces the widget');
    const replacement = browser.document.createElement('div');
    const cleanupReplacement = launcher.render({ el: replacement });
    browser.flushTimers();
    assert.equal(browser.overlay(), overlay);
    assert.equal(browser.document.documentElement.style.overflow, 'hidden');
    assert.equal(browser.listeners.get('click')?.size, 1);
    assert.equal(browser.requests.length, 1, 'remounting must not reload the presentation');
    cleanupReplacement();
    browser.flushTimers();
    assert.equal(browser.overlay(), null);
    assert.equal(browser.document.documentElement.style.overflow, '');
    assert.equal(browser.document.activeElement, anchor);
    assert.equal(browser.listeners.get('click')?.size, 0);
  });
});

test('a new chapter mount closes the previous chapter presentation', async () => {
  await withBrowser('http://localhost:3000/modeling-controlled-systems', async (browser) => {
    const cleanupFirst = launcher.render({ el: browser.el });
    await browser.click('http://localhost:3100/presenter.html');
    cleanupFirst();
    browser.window.location = new URL('http://localhost:3000/stochastic-dp');
    const cleanupSecond = launcher.render({ el: browser.document.createElement('div') });
    browser.flushTimers();
    assert.equal(browser.overlay(), null);
    assert.equal(browser.listeners.get('click')?.size, 1);
    await browser.click('http://localhost:3100/presenter.html');
    assert.equal(browser.overlay().dataset.page, browser.window.location.href);
    cleanupSecond();
    browser.flushTimers();
  });
});
