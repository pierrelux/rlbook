window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    packages: {'[+]': ['action']},
    macros: {
      texttippy: ['\\action{#1}{tippytooltip=#2}', 2]
    }
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      console.log('MathJax is ready');
      MathJax.startup.promise.then(() => {
        console.log('MathJax rendering complete');
        if (typeof tippy !== 'undefined') {
          tippy('[data-tippy-content]', {
            // Tippy.js options here
          });
          console.log('Tippy initialized');
        } else {
          console.error('Tippy.js is not loaded');
        }
      });
    }
  }
};