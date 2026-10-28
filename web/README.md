# Transaction demo

The demo is plain HTML, CSS, and JavaScript. It does not require a frontend dependency installation or build step. The page, styles, and browser modules live directly in `web/`; contract tests live in `web/tests/`.

Start the API using the repository's runtime instructions, then open **http://localhost:8000/demo/**. FastAPI serves the web assets at `/demo/`. Model artifacts stay in the separate `models/` directory.

The demo loads the required feature names and order from `GET /model-info`. Enter an ordered JSON array of raw numeric features, or choose **Fill synthetic example**. The example contains invented zeros and an Amount of 100 when that feature is present. It has no expected outcome. Every displayed prediction and score comes from `POST /predict`.

The model probability is not proof of fraud or a measured accuracy figure. Explanation status is shown separately. **Check explanation** retrieves `GET /explain/{transaction_id}`; pending, absent, and failed results are not presented as completed explanations. This button does not automatically poll.

Use a unique transaction ID for each request. The demo generates one when supported by the browser; otherwise enter one before scoring. If a response is lost or times out, the request may still have been processed. Keep the ID, enter it in the transaction ID field if needed, and use **Check explanation** before deciding what to do next. The lookup shows the ID and selected API and remains available without a received prediction. The selected API may redirect the request, so it is not proof of which server responded. A reused ID returns HTTP 409. A missing explanation does not prove that the request was not processed.

The optional API URL supports a separate HTTP(S) API, but browser cross-origin access must be allowed by that server. Same-origin `/demo/` is the recommended setup. No API URL or transaction is saved in browser storage.

## Checks

From the repository root, with Node.js installed:

```sh
node --test web/tests/*.test.mjs
node --check web/app.mjs
node --check web/api-client.mjs
```

Tests cover feature and response contracts, probability formatting, URL validation, and HTTP error handling with a temporary local test server. These tests do not prove model accuracy, the deployed image, worker completion, or browser integration. Verify those against the running stack separately.
