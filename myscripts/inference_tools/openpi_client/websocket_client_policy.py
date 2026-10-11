"""Local compatibility wrapper; connects only to a LingBot RTC server."""

from ..websocket_client_policy import WebsocketClientPolicy as _Client


class WebsocketClientPolicy(_Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metadata = self.get_server_metadata()
        required = {"model_backend": "lingbot-vla-v2", "action_horizon": 50,
                    "physical_action_dim": 20, "rtc_capable": True}
        mismatches = {key: (metadata.get(key), expected) for key, expected in required.items()
                      if metadata.get(key) != expected}
        if mismatches:
            raise RuntimeError(f"Refusing non-LingBot/incompatible server: {mismatches}")
