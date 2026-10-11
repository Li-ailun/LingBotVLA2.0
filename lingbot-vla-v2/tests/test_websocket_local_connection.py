from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_default_client_connects_directly_when_http_proxy_is_configured() -> None:
    program = textwrap.dedent(
        """
        import threading
        from deploy.msgpack_numpy import Packer
        from deploy.websocket_client_policy import WebsocketClientPolicy
        from websockets.sync.server import serve

        packer = Packer()
        def handler(ws):
            ws.send(packer.pack({"ready": True}))
            ws.recv()

        with serve(handler, "127.0.0.1", 0) as server:
            port = server.socket.getsockname()[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = WebsocketClientPolicy(port=port)
            assert client.get_server_metadata() == {"ready": True}
        """
    )
    env = os.environ.copy()
    env.update({
        "ws_proxy": "http://127.0.0.1:9",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "no_proxy": "localhost,127.0.0.1,::1",
    })
    result = subprocess.run(
        [sys.executable, "-c", program], env=env,
        text=True, capture_output=True, timeout=5, check=False,
    )
    assert result.returncode == 0, result.stderr
