import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 1,
  duration: '1s',
};

export default function () {
  const url = `${__ENV.BASE_URL}/v1/move`;
  const payload = JSON.stringify({});
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(url, payload, params);
  check(res, { 'status was 200': (r) => r.status === 200 });
}
