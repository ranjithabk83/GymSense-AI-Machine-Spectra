/**
 * GYMSENSE AI - Progressive Web App Handler
 * Student: Ranjitha B K | USN: 1SB24AI041
 */

let deferredInstallPrompt = null;

// Register Service Worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then((reg) => {
        console.log('[PWA] Service Worker registered successfully with scope:', reg.scope);
      })
      .catch((err) => {
        console.warn('[PWA] Service Worker registration failed:', err);
      });
  });
}

// Capture BeforeInstallPrompt
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  const pwaInstallBtn = document.getElementById('pwa-install-btn');
  if (pwaInstallBtn) {
    pwaInstallBtn.style.display = 'flex';
  }
});

function triggerPWAInstall() {
  if (deferredInstallPrompt) {
    deferredInstallPrompt.prompt();
    deferredInstallPrompt.userChoice.then((choiceResult) => {
      if (choiceResult.outcome === 'accepted') {
        console.log('[PWA] User accepted the installation');
      }
      deferredInstallPrompt = null;
      const pwaInstallBtn = document.getElementById('pwa-install-btn');
      if (pwaInstallBtn) pwaInstallBtn.style.display = 'none';
    });
  } else {
    alert('GymSense AI is already installed or ready in your browser menu (Add to Home Screen).');
  }
}

window.triggerPWAInstall = triggerPWAInstall;
