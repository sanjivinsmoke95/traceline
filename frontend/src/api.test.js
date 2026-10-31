import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_DOWN, api } from './api';

afterEach(() => vi.restoreAllMocks());

describe('api when the backend is not running', () => {
  it('explains a refused connection', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(api.cases()).rejects.toMatchObject({ code: 'api_unreachable', message: API_DOWN });
  });
  it('explains the dev proxy\'s empty 500', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500, headers: { 'content-type': 'text/plain' } })));
    await expect(api.cases()).rejects.toMatchObject({ code: 'api_unreachable' });
  });
  it('keeps a real API error message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: 'CASE_NOT_FOUND', message: 'No such case' } }), { status: 404, headers: { 'content-type': 'application/json' } })));
    await expect(api.case('x')).rejects.toMatchObject({ status: 404, code: 'CASE_NOT_FOUND', message: 'No such case' });
  });
});
