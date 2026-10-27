# Transaction demo

The demo is plain HTML, CSS, and JavaScript. It does not require a frontend dependency installation or build step. The page, styles, and browser modules live directly in `web/`; contract tests live in `web/tests/`.

Start the API using the repository's runtime instructions, then open **http://localhost:8000/demo/**. FastAPI serves the web assets at `/demo/`. Model artifacts stay in the separate `models/` directory.

The demo loads the required feature names and order from `GET /model-info`. Enter an ordered JSON array of raw numeric features, or choose **Fill synthetic example**. The example contains invented zeros and an Amount of 100 when that feature is present. It has no expected outcome. Every displayed prediction and score comes from `POST /predict`.

The model probability is not proof of fraud or a measured accuracy figure. Explanation status is shown separately. **Check explanation** retrieves `GET /explain/{transaction_id}`; pending, absent, and failed results are not presented as completed explanations. This button does not automatically poll.

Use a unique transaction ID for each request. The demo generates one when supported by the browser; otherwise the API generates it. If a request times out, it may still have been processed. Keep the displayed ID and check its explanation before resubmitting. A reused ID returns HTTP 409.

When a valid submitted ID is known, **Check explanation** remains available after a failed or timed-out scoring response. It queries that ID on the original API connection without resubmitting the transaction. The prediction stays hidden; a failed response does not establish whether the request was processed or saved. Editing the features, transaction ID or API URL, or reconnecting to the model, clears the score and lookup context.

If the ID was left blank and the browser cannot generate a UUID, a failed response leaves no known submitted ID for this demo to query. It does not invent a lookup route. A validated successful response uses its returned ID for the normal explanation flow when that ID meets the lookup rules; otherwise no lookup is offered.

The optional API URL supports a separate HTTP(S) API, but browser cross-origin access must be allowed by that server. Same-origin `/demo/` is the recommended setup. No API URL or transaction is saved in browser storage.

## Checks

From the repository root, with Node.js installed:

```sh
node --test web/tests/*.test.mjs
node --check web/app.mjs
node --check web/api-client.mjs
```

Tests cover feature and response contracts, probability formatting, URL validation, and HTTP error handling with a temporary local test server. These tests do not prove model accuracy, the deployed image, worker completion, or browser integration. Verify those against the running stack separately.

The recovery regressions execute the actual browser modules with a minimal DOM and in-memory responses, including ambiguous confirmation, timeout, stale results and explicit status checks. Plain `node --test` starts a bounded, reaped child with the VM module flag when needed; it requires no global Node flag or dependency. This source-state coverage does not replace fresh browser layout and interaction checks.
