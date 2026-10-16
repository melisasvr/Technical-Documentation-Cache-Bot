# Acme API Guide

## Authentication
All requests need an `Authorization: Bearer <token>` header. Tokens expire after 60 minutes.
Refresh a token by calling `POST /v1/auth/refresh` with your refresh token.

## Rate Limits
The default limit is 100 requests per minute per API key. Exceeding it returns HTTP 429.
The `Retry-After` header tells you how many seconds to wait.

## Pagination
List endpoints accept `limit` (max 200, default 50) and `cursor`. The response contains
`next_cursor`; when it is null there are no more results.

## Errors
Errors return JSON: `{"error": {"code": "invalid_request", "message": "..."}}`.
HTTP 5xx errors are safe to retry with exponential backoff.
