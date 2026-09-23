# Frontend

React 19 + Vite dashboard for the age progression API. See the [root README](../README.md) for the full project.

```bash
npm install
npm run dev       # http://localhost:5173
npm run lint
npm run build     # production bundle in dist/
```

The API base URL defaults to `http://localhost:8000`. Override it with `VITE_API_URL` at dev or build time:

```bash
VITE_API_URL=http://192.168.1.20:8000 npm run build
```

Serving the dashboard from an origin other than `localhost`/`127.0.0.1` on ports 5173 or 4173 also requires adding that origin to the CORS list in `backend/main.py`.
