# Iframe Modal Component Usage Guide

The `iframe-modal.js` component provides a reusable way to embed interactive demos with expandable full-screen modal views.

## Basic Usage

### Method 1: JavaScript API

```html
<script src="_static/iframe-modal.js"></script>
<div id="my-demo-container"></div>
<script>
createIframeModal({
  containerId: 'my-demo-container',
  iframeSrc: '_static/my-demo.html',
  title: 'My Interactive Demo',
  aspectRatio: '200%',
  maxWidth: '1400px',
  maxHeight: '900px'
});
</script>
```

### Method 2: Data Attributes (Auto-initialization)

```html
<script src="_static/iframe-modal.js"></script>
<div id="my-demo-container" 
     data-iframe-modal
     data-iframe-src="_static/my-demo.html"
     data-title="My Interactive Demo"
     data-aspect-ratio="200%"
     data-max-width="1400px"
     data-max-height="900px">
</div>
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `containerId` | **required** | ID of the container element |
| `iframeSrc` | **required** | Path to the iframe content |
| `title` | `'Demo'` | Title for accessibility and modal header |
| `aspectRatio` | `'200%'` | Height as percentage of width (e.g., '200%' = 2:1 ratio) |
| `maxWidth` | `'1400px'` | Maximum width of expanded modal |
| `maxHeight` | `'900px'` | Maximum height of expanded modal |
| `borderRadius` | `'10px'` | Border radius for embedded iframe |

## Return Object Methods

The `createIframeModal()` function returns an object with utility methods:

```javascript
const modal = createIframeModal({...});

modal.open();     // Open the modal
modal.close();    // Close the modal  
modal.toggle();   // Toggle modal state

// Access DOM elements
modal.getModal();           // Get modal overlay element
modal.getIframe();          // Get embedded iframe
modal.getExpandedIframe();  // Get expanded modal iframe
```

## Features

- **Responsive**: Embedded iframe scales with container
- **Accessible**: Keyboard navigation (Escape to close)
- **Reusable**: Multiple modals on same page supported
- **No conflicts**: Auto-generated unique IDs
- **Click outside to close**: Modal closes when clicking overlay
- **Visual feedback**: Button text changes when expanded

## Examples

### Standard Aspect Ratio
```html
<script src="_static/iframe-modal.js"></script>
<div id="standard-demo"></div>
<script>
createIframeModal({
  containerId: 'standard-demo',
  iframeSrc: '_static/demo.html',
  title: 'Standard Demo'
});
</script>
```

### Wide Format
```html
<script src="_static/iframe-modal.js"></script>
<div id="wide-demo"></div>
<script>
createIframeModal({
  containerId: 'wide-demo',
  iframeSrc: '_static/wide-demo.html',
  title: 'Wide Format Demo',
  aspectRatio: '56.25%',  // 16:9 ratio
  maxWidth: '1600px'
});
</script>
```

### Compact Format
```html
<script src="_static/iframe-modal.js"></script>
<div id="compact-demo"></div>
<script>
createIframeModal({
  containerId: 'compact-demo',
  iframeSrc: '_static/compact-demo.html',
  title: 'Compact Demo',
  aspectRatio: '100%',  // 1:1 square ratio
  maxWidth: '800px',
  maxHeight: '800px'
});
</script>
``` 