/*
  GYMSENSE AI - Progressive Web App Service Worker
  Student: Ranjitha B K | USN: 1SB24AI041
*/

const CACHE_NAME = 'gymsense-ai-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/manifest.json',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/js/sensors.js',
  '/static/js/workout.js',
  '/static/js/charts.js',
  '/static/js/avatarEngine.js',
  '/static/js/history.js',
  '/static/js/pwa.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800;900&display=swap'
];

// Install Event
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Pre-caching offline application shell');
      return cache.addAll(ASSETS_TO_CACHE).catch((err) => {
        console.warn('[ServiceWorker] Non-critical pre-cache warning:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

// Activate Event
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing legacy cache:', name);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Strategy: Network First for API, Cache First for Static Assets
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Dynamic API calls (Predictions, model info, etc.) -> Network First
  if (url.pathname.startsWith('/predict') || url.pathname.startsWith('/model-info') || url.pathname.startsWith('/health') || url.pathname.startsWith('/training-data')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        return new Response(JSON.stringify({ error: 'Offline mode active' }), {
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }

  // Static Assets -> Cache First with Network Fallback
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        return networkResponse;
      });
    }).catch(() => {
      // Offline fallback for HTML
      if (event.request.headers.get('accept')?.includes('text/html')) {
        return caches.match('/');
      }
    })
  );
});
