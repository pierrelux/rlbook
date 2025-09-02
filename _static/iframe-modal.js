/**
 * Reusable Iframe Modal Component
 * 
 * Usage:
 * 1. Include this script in your HTML
 * 2. Call createIframeModal(config) to create a modal
 * 
 * Example:
 * createIframeModal({
 *   containerId: 'my-container',
 *   iframeSrc: 'path/to/demo.html',
 *   title: 'My Demo',
 *   aspectRatio: '200%', // height as percentage of width
 *   maxWidth: '1400px',
 *   maxHeight: '900px'
 * });
 */

function createIframeModal(config) {
    const {
        containerId,
        iframeSrc,
        title = 'Demo',
        aspectRatio = '200%',
        maxWidth = '1400px',
        maxHeight = '900px',
        borderRadius = '10px'
    } = config;

    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container with id "${containerId}" not found`);
        return;
    }

    // Generate unique IDs to avoid conflicts
    const modalId = `modal-${containerId}`;
    const iframeId = `iframe-${containerId}`;
    const expandBtnId = `expand-btn-${containerId}`;
    const expandedIframeId = `expanded-iframe-${containerId}`;

    // Create the embedded iframe with expand button
    container.innerHTML = `
        <div style="position: relative; width: 100%; height: 0; padding-bottom: ${aspectRatio}; overflow: hidden; border-radius: ${borderRadius};">
            <div style="position: absolute; top: 10px; right: 10px; z-index: 1000;">
                <button id="${expandBtnId}" 
                        style="background: rgba(0,0,0,0.7); color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                    </svg>
                    Expand View
                </button>
            </div>
            <iframe id="${iframeId}" src="${iframeSrc}" 
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; border-radius: ${borderRadius};"
                    title="${title}">
            </iframe>
        </div>
    `;

    // Create the modal overlay (append to body)
    const modalOverlay = document.createElement('div');
    modalOverlay.id = modalId;
    modalOverlay.style.cssText = `
        display: none; 
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
        background: rgba(0,0,0,0.8); 
        z-index: 10000; 
        backdrop-filter: blur(2px);
    `;

    modalOverlay.innerHTML = `
        <div style="position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; padding: 20px;">
            <div style="position: relative; width: 90%; height: 90%; max-width: ${maxWidth}; max-height: ${maxHeight}; background: white; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;">
                <!-- Close button -->
                <button id="close-${modalId}" 
                        style="position: absolute; top: 15px; right: 15px; background: rgba(0,0,0,0.7); color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; z-index: 10001; display: flex; align-items: center; gap: 6px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                    </svg>
                    Close
                </button>
                <!-- Expanded iframe -->
                <iframe id="${expandedIframeId}" src="${iframeSrc}" 
                        style="width: 100%; height: 100%; border: none; border-radius: 12px;"
                        title="${title} - Expanded View">
                </iframe>
            </div>
        </div>
    `;

    document.body.appendChild(modalOverlay);

    // Toggle function
    function toggleModal() {
        const modal = document.getElementById(modalId);
        const expandBtn = document.getElementById(expandBtnId);
        
        if (modal.style.display === 'none') {
            // Show expanded view
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
            
            expandBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>
                </svg>
                Expanded
            `;
        } else {
            // Hide expanded view
            modal.style.display = 'none';
            document.body.style.overflow = '';
            
            expandBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                </svg>
                Expand View
            `;
        }
    }

    // Event listeners
    document.getElementById(expandBtnId).addEventListener('click', toggleModal);
    document.getElementById(`close-${modalId}`).addEventListener('click', toggleModal);

    // Close modal when clicking outside the content area
    modalOverlay.addEventListener('click', function(e) {
        if (e.target === modalOverlay) {
            toggleModal();
        }
    });

    // Global escape key handler (only add once)
    if (!window.iframeModalEscapeAdded) {
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                // Close any open modal
                const openModals = document.querySelectorAll('[id^="modal-"]:not([style*="display: none"])');
                if (openModals.length > 0) {
                    openModals[0].style.display = 'none';
                    document.body.style.overflow = '';
                    
                    // Reset expand button text
                    const expandBtns = document.querySelectorAll('[id^="expand-btn-"]');
                    expandBtns.forEach(btn => {
                        btn.innerHTML = `
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                            </svg>
                            Expand View
                        `;
                    });
                }
            }
        });
        window.iframeModalEscapeAdded = true;
    }

    // Return object with utility methods
    return {
        open: () => toggleModal(),
        close: () => {
            const modal = document.getElementById(modalId);
            if (modal.style.display !== 'none') {
                toggleModal();
            }
        },
        toggle: toggleModal,
        getModal: () => document.getElementById(modalId),
        getIframe: () => document.getElementById(iframeId),
        getExpandedIframe: () => document.getElementById(expandedIframeId)
    };
}

// Auto-initialize any elements with data-iframe-modal attribute
document.addEventListener('DOMContentLoaded', function() {
    const autoElements = document.querySelectorAll('[data-iframe-modal]');
    autoElements.forEach(element => {
        const config = {
            containerId: element.id,
            iframeSrc: element.dataset.iframeSrc,
            title: element.dataset.title || 'Demo',
            aspectRatio: element.dataset.aspectRatio || '200%',
            maxWidth: element.dataset.maxWidth || '1400px',
            maxHeight: element.dataset.maxHeight || '900px',
            borderRadius: element.dataset.borderRadius || '10px'
        };
        createIframeModal(config);
    });
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createIframeModal };
} 