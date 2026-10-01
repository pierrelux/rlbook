export default {
  name: 'Same-origin spotlight presenter',
  transforms: [{
    name: 'presenter-launcher',
    stage: 'document',
    plugin: () => (tree) => {
      if (process.argv.some((arg) => ['pdf', 'tex', 'typst', 'docx'].includes(arg.replace(/^--/, '')))) return;
      // A leading slash means the project root to MyST's asset resolver.
      tree.children.push({
        type: 'anywidget',
        id: 'rl-presenter-launcher',
        esm: '/_static/presenter-launcher.mjs',
        model: {},
      });
    },
  }],
};
