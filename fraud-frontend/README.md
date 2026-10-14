# Transaction demo

The demo is plain HTML, CSS, and JavaScript. The previous checked-in frontend had a Vite entry page but no package manifest or React source. This recovery does not require a frontend dependency installation or build step.

Start the API using the repository's runtime instructions, then open **http://localhost:8000/demo/**. FastAPI serves only `demo/`; the duplicate model binaries in this directory are not web assets.

The demo loads the required feature names and order from `GET /model-info`. Enter an ordered JSON array of raw numeric features, or choose **Fill synthetic example**. The example contains invented zeros and an Amount of 100 when that feature is present. It has no expected outcome. Every displayed prediction and score comes from `POST /predict`.

The model probability is not proof of fraud or a measured accuracy figure. Explanation status is shown separately. **Check explanation** retrieves `GET /explain/{transaction_id}`; pending, absent, and failed results are not presented as completed explanations. This button does not automatically poll.

Use a unique transaction ID for each request. The demo generates one when supported by the browser; otherwise the API generates it. If a request times out, it may still have been processed. Keep the displayed ID and check its explanation before resubmitting. A reused ID returns HTTP 409.

The optional API URL supports a separate HTTP(S) API, but browser cross-origin access must be allowed by that server. Same-origin `/demo/` is the recommended setup. No API URL or transaction is saved in browser storage.

## Checks

From the repository root, with Node.js installed:

```sh
node --test fraud-frontend/tests/*.test.mjs
node --check fraud-frontend/demo/app.mjs
node --check fraud-frontend/demo/api-client.mjs
```

Tests cover feature and response contracts, probability formatting, URL validation, and HTTP error handling with a temporary local test server. These tests do not prove model accuracy, the deployed image, worker completion, or browser integration. Verify those against the running stack separately.
