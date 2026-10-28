# Transaction demo

The demo is plain HTML, CSS, and JavaScript. It does not require a frontend dependency installation or build step. The page, styles, and browser modules live directly in `web/`; contract tests live in `web/tests/`.

Start the API using the repository's runtime instructions, then open **http://localhost:8000/demo/**. FastAPI serves the web assets at `/demo/`. Model artifacts stay in the separate `models/` directory.

The demo loads the required feature names and order from `GET /model-info`. Enter an ordered JSON array of raw numeric features, or choose **Fill synthetic example**. The example contains invented zeros and an Amount of 100 when that feature is present. It has no expected outcome. Every displayed prediction and score comes from `POST /predict`.

The model probability is not proof of fraud or a measured accuracy figure. Explanation status is shown separately. **Check explanation** retrieves `GET /explain/{transaction_id}`; pending, absent, and failed results are not presented as completed explanations. This button does not automatically poll.

Use a unique transaction ID for each different transaction. The demo uses browser `randomUUID`, or secure random bytes from `getRandomValues` when needed. If neither generator works, enter an ID before scoring. If a request times out or the API reports a conflict, it may still have been processed. Check the saved explanation before deciding whether to retry. Use a new ID only for a different transaction.

When a valid submitted ID is known, **Check explanation** remains available after a failed or timed-out scoring response. It queries that ID on the original API connection without resubmitting the transaction. A prediction naming a different ID is rejected, and an explanation with a missing or different ID is not displayed. Structured errors carrying an ID or a recognized transaction status must also identify the same ID; other errors remain visible. The prediction stays hidden; a failed response does not establish whether the request was processed or saved. Editing the features, transaction ID or API URL, or reconnecting to the model, clears the score and lookup context.

The application does not initiate a second prediction automatically. A browser or network may still make repeated HTTP attempts after a connection failure. The database's unique explanation ID can reject a later duplicate insert, but scoring occurs before that check, so the same ID does not guarantee exactly one inference or one server arrival.

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
