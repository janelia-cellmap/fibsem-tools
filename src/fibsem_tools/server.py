import contextlib
import random
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer, test

DEFAULT_PORT = 5000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_DIRECTORY = "."


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        return super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()


def serve(*, port: int, bind: str, directory: str) -> None:
    """
    Start up a simple static file server.
    Adapated from the source code of http.server in the stdlib.
    """

    attempts = 11
    attempt = 1

    handler_class = CORSRequestHandler

    # it's ugly to define a class inside a function, but this appears necessary due
    # to the need for the directory variable to be passed to DualStackServer.finish_request
    class DualStackServer(ThreadingHTTPServer):
        def server_bind(self) -> None:
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address) -> None:
            self.RequestHandlerClass(request, client_address, self, directory=directory)

    while attempt < attempts:
        try:
            test(
                HandlerClass=handler_class,
                ServerClass=DualStackServer,
                port=port,
                bind=bind,
            )

        except OSError:
            port = random.randint(8000, 65535)
            attempts += 1

    msg = f"Failed to get a port after {attempt} attempts. Closing."
    raise RuntimeError(msg)
