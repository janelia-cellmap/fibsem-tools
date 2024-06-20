import click
from fibsem_tools.server import DEFAULT_DIRECTORY, DEFAULT_HOST, DEFAULT_PORT, serve


@click.command
@click.option(
    "-d",
    "--dir",
    type=click.Path(exists=True, file_okay=False),
    default=DEFAULT_DIRECTORY,
    help="serve this directory " "(default: current directory)",
)
@click.option(
    "-b",
    "--bind",
    type=click.STRING,
    default=DEFAULT_HOST,
    help="bind to this address " f"(default: {DEFAULT_HOST})",
)
@click.option(
    "-p",
    "--port",
    type=click.INT,
    default=DEFAULT_PORT,
    help="bind to this port " f"(default: {DEFAULT_PORT}",
)
def serve_cli(dir: str, port: int, bind: str):
    """
    Start up a simple static file server with permissive CORS headers.
    """
    serve(port=port, directory=dir, bind=bind)
