from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM


class InterfaceLLM:
    """Use the configured LLM service and fail explicitly on backend errors."""

    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local,
                 llm_local_url, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url

        if llm_use_local:
            if not llm_local_url or llm_local_url == "xxx":
                raise ValueError("Configure a real local LLM service URL.")
            self.interface_llm = InterfaceLocalLLM(llm_local_url)
        else:
            if any(not value or value == "xxx"
                   for value in (api_endpoint, api_key, model_LLM)):
                raise ValueError("Configure the LLM endpoint, API key, and model.")
            self.interface_llm = InterfaceAPI(
                api_endpoint, api_key, model_LLM, debug_mode
            )

        # Preserve the original connectivity check using an actual API request.
        self.get_response("1+1=?")

    def get_response(self, prompt_content):
        try:
            response = self.interface_llm.get_response(prompt_content)
        except Exception as exc:
            raise RuntimeError("Configured LLM service failed.") from exc
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Configured LLM service returned an empty response.")
        return response