import http.client
import json
from urllib.parse import urlparse


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 2
        self.request_timeout_sec = 20
        self.host, self.path = self._normalize_endpoint(api_endpoint)

    def _normalize_endpoint(self, endpoint):
        endpoint = endpoint.strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            parsed = urlparse(endpoint)
            host = parsed.netloc
            path = parsed.path.rstrip("/")
            if path.endswith("/v1/chat/completions"):
                req_path = path
            elif path.endswith("/v1"):
                req_path = path + "/chat/completions"
            elif path == "":
                req_path = "/v1/chat/completions"
            else:
                req_path = path + "/v1/chat/completions"
            return host, req_path
        return endpoint, "/v1/chat/completions"

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }
        
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            conn = None
            try:
                conn = http.client.HTTPSConnection(self.host, timeout=self.request_timeout_sec)
                conn.request("POST", self.path, payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            

        return response
