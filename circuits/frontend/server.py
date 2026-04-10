import atexit
import functools
import gzip
import http.server
import json
import logging
import os
import socketserver
import threading
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
logger.propagate = False

DEFAULT_FRONTEND_DIR = Path(__file__).parent / "assets"


class ListHandler(logging.Handler):
    """Handler that appends log records to a list."""

    def __init__(self, log_list: list[str]):
        super().__init__()
        self.log_list = log_list

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.log_list.append(msg)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class CircuitGraphHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, frontend_dir: str | Path, data_dir: str | Path, **kwargs):  # type: ignore[no-untyped-def]
        self.data_dir = str(data_dir)
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        message = format % args
        logger.info(
            "%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), message)
        )

    def do_GET(self) -> None:
        try:
            self._do_GET()
        except Exception as e:
            logger.exception(f"Error handling GET request: {e}")
            self.send_response(500)
            self.end_headers()

    def _handle_neuron_exemplars(self) -> None:
        """Fetch neuron exemplar data from the Modal neuron-data-server."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        layer = params.get("layer", [None])[0]
        neuron = params.get("neuron", [None])[0]

        if layer is None or neuron is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "missing layer or neuron param"}')
            return

        sign = params.get("sign", ["%2B"])[0]
        sign_encoded = "-" if sign == "-" else "%2B"
        modal_url = (
            f"https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"
            f"?layer={layer}&neuron={neuron}&sign={sign_encoded}"
        )

        try:
            req = urllib.request.Request(modal_url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception:
            logger.exception(f"Failed to fetch neuron exemplars for L{layer}N{neuron}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps({"isDead": True}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Transform Modal response into frontend format.
        # Modal returns {"exemplar_tokens": [[tokens, activations], ...], ...}
        # May also return {"detail": "..."} for missing neurons.
        if "detail" in raw and "exemplar_tokens" not in raw:
            body = json.dumps({"isDead": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        exemplar_tokens = raw.get("exemplar_tokens", [])
        examples = []
        global_max = 0.0
        for item in exemplar_tokens:
            tokens, acts = item[0], item[1]
            if not tokens:
                continue
            max_act = max(acts) if acts else 0.0
            if max_act > global_max:
                global_max = max_act
            train_idx = acts.index(max_act) if acts else 0
            examples.append(
                {"tokens": tokens, "tokens_acts_list": acts, "train_token_ind": train_idx}
            )

        result = {
            "act_min": 0,
            "act_max": global_max if global_max > 0 else 1.0,
            "examples_quantiles": [{"quantile_name": "Max activating", "examples": examples}],
            "top_logits": [],
            "bottom_logits": [],
        }

        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _do_GET(self) -> None:
        logger.info(f"Received request for {self.path}")

        if self.path.startswith("/api/neuron_exemplars"):
            self._handle_neuron_exemplars()
            return

        if self.path.startswith(("/data/", "/graph_data/")):
            if self.path.startswith("/data/"):
                rel_path = self.path[len("/data/") :].split("?")[0]
            else:
                rel_path = self.path[len("/graph_data/") :].split("?")[0]

            local_path = os.path.join(self.data_dir, rel_path)

            logger.info(f"Rewritten path to {local_path}.")
            if not os.path.exists(local_path):
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            with open(local_path, "rb") as f:
                content = f.read()

            if len(content) > 1024**2:
                content = gzip.compress(content, compresslevel=3)
                self.send_header("Content-Encoding", "gzip")

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        super().do_GET()

    def do_POST(self) -> None:
        if not self.path.startswith("/save_graph/"):
            self.send_response(404)
            return

        try:
            parts = self.path.split("?")[0].strip("/").split("/")
            slug = parts[-1]

            logger.info(f"Saving graph for {slug}")

            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            save_path = os.path.join(self.data_dir, f"{slug}.json")

            with open(save_path) as f:
                graph = json.load(f)
                graph["qParams"] = data["qParams"]

            with open(save_path, "w") as f:
                json.dump(graph, f, indent=2)

            self.send_response(200)
            self.end_headers()
            logger.info(f"Graph saved: {save_path}")

        except Exception as e:
            logger.exception(f"Error saving graph: {e}")
            self.send_response(500)
            self.end_headers()


class Server:
    def __init__(self, httpd: ReusableTCPServer, server_thread: threading.Thread) -> None:
        self.httpd = httpd
        self.server_thread = server_thread
        self.logs: list[str] = []
        self._stopped = False

        self.log_handler = ListHandler(self.logs)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)
        atexit.register(self.stop)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        logger.info("Stopping server...")

        try:
            self.httpd.socket.close()
        except Exception as e:
            logger.debug(f"Error closing socket: {e}")

        shutdown_thread = threading.Thread(target=self.httpd.shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()

        shutdown_thread.join(timeout=5)
        self.server_thread.join(timeout=5)

        try:
            self.httpd.server_close()
        except Exception as e:
            logger.debug(f"Error during server_close: {e}")

        logger.info("Server stopped")

        logger.removeHandler(self.log_handler)
        atexit.unregister(self.stop)

    def get_logs(self) -> list[str]:
        """Return the current log messages."""
        return self.logs


def serve(
    data_dir: str | Path,
    frontend_dir: str | Path | None = None,
    port: int = 8032,
) -> Server:
    """Start a local HTTP server in a separate thread.

    Args:
        data_dir: Directory for local graph data.
        frontend_dir: Directory containing frontend files. Defaults to bundled assets.
        port: Port to serve on. Defaults to 8032.

    Returns:
        Server object with a stop() method to shut down the server.
    """
    frontend_dir = Path(frontend_dir).resolve() if frontend_dir else DEFAULT_FRONTEND_DIR

    frontend_dir_path = Path(frontend_dir)
    if not frontend_dir_path.exists() or not frontend_dir_path.is_dir():
        raise ValueError(f"Got frontend dir {frontend_dir} but this is not a valid directory")

    logger.info(f"Serving files from: {frontend_dir}")

    handler = functools.partial(CircuitGraphHandler, frontend_dir=frontend_dir, data_dir=data_dir)

    httpd = ReusableTCPServer(("", port), handler)

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    logger.info(f"Serving at http://localhost:{port}")
    logger.info(f"Serving data from: {data_dir}")

    return Server(httpd, server_thread)
