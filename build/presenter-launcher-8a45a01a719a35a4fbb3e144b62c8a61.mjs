// Run in the chapter's origin. MyST serves this module and presenter.html
// from a separate asset server during development.
const key = Symbol.for('rlbook.presenter.launcher');

function createController(doc, host) {
  const mounts = new Set();
  const currentElement = () => [...mounts].at(-1)?.el;
  let overlay;
  let cancelled = false;
  let pending = false;
  let disposalTimer;
  let restoreOverflow;
  let returnFocus;

  const close = () => {
    if (!overlay) return;
    overlay.remove();
    overlay = undefined;
    doc.documentElement.style.overflow = restoreOverflow;
    returnFocus?.focus({ preventScroll: true });
  };

  const launch = async (event) => {
    const link = event.target.closest?.('a[href]');
    if (!link || !/^presenter(?:-[a-f0-9]+)?\.html$/.test(new URL(link.href).pathname.split('/').pop())) return;
    event.preventDefault();
    event.stopPropagation();
    if (pending || overlay) return;

    // Capture before awaiting: the reader may change chapters while fetching.
    const page = host.location.href;
    pending = true;
    currentElement()?.replaceChildren();
    try {
      const response = await host.fetch(link.href);
      if (!response.ok) throw new Error(`Presenter request failed (${response.status})`);
      const html = await response.text();
      if (cancelled || mounts.size === 0 || host.location.href !== page) return;
      returnFocus = doc.activeElement;
      restoreOverflow = doc.documentElement.style.overflow;
      overlay = doc.createElement('iframe');
      overlay.id = 'rl-presenter-overlay';
      overlay.title = 'Recorded spotlight presenter';
      overlay.dataset.page = page;
      overlay.rlPresenterExit = close;
      Object.assign(overlay.style, {
        position: 'fixed', inset: '0', width: '100%', height: '100%',
        border: '0', zIndex: '2147483647', background: 'white',
      });
      // srcdoc inherits this chapter's origin, so the recorder can access its
      // nested notes frame even when the HTML was fetched from another port.
      overlay.srcdoc = html;
      doc.documentElement.style.overflow = 'hidden';
      doc.body.append(overlay);
      overlay.focus();
    } catch (error) {
      if (cancelled || mounts.size === 0 || host.location.href !== page) return;
      const message = doc.createElement('p');
      message.setAttribute('role', 'alert');
      message.textContent = 'The presenter could not load. Please try Present again.';
      currentElement()?.replaceChildren(message);
      host.console.error(error);
    } finally {
      pending = false;
    }
  };

  const dispose = () => {
    cancelled = true;
    doc.removeEventListener('click', launch, true);
    close();
    if (host[key] === controller) delete host[key];
  };
  const controller = {
    retain(el) {
      host.clearTimeout(disposalTimer);
      const mount = { el };
      mounts.add(mount);
      if (overlay && overlay.dataset.page !== host.location.href) close();
      return () => {
        if (!mounts.delete(mount) || mounts.size > 0) return;
        // MyST can remount the widget when chapter content updates. Let its
        // replacement retain the controller before disposing the active deck.
        disposalTimer = host.setTimeout(dispose, 0);
      };
    },
  };
  doc.addEventListener('click', launch, true);
  return controller;
}

export function render({ el }) {
  const doc = el.ownerDocument;
  const host = doc.defaultView;
  if (host !== host.top) return;

  host[key] ??= createController(doc, host);
  return host[key].retain(el);
}

export default { render };
