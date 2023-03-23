from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer, test
import socket
import random

PORT = 5000
HOST = "0.0.0.0"
DIRECTORY = "."


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        return super(CORSRequestHandler, self).end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def serve(port, bind, directory):
    """
    Server code adapated from the source code of http.server in the stdlib
    """
    attempts = 11
    attempt = 1

    handler_class = CORSRequestHandler

    class DualStackServer(ThreadingHTTPServer):
        def server_bind(self):
            with contextlib.suppress(Exception):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            return super().server_bind()

        def finish_request(self, request, client_address):
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

    raise RuntimeError(f"Failed to get a port after {attempt} attempts. Closing.")


if __name__ == "__main__":
    import argparse
    import contextlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bind",
        metavar="ADDRESS",
        default=HOST,
        help="bind to this address " "(default: 0.0.0.0)",
    )
    parser.add_argument(
        "directory",
        default=DIRECTORY,
        help="serve this directory " "(default: current directory)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=8000,
        type=int,
        nargs="?",
        help="bind to this port " "(default: %(default)s)",
    )

    args = parser.parse_args()

    serve(args.port, args.bind, args.directory)
