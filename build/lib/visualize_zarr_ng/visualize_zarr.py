#!/usr/bin/env python3
# Usage:
#   python ng_zarr_demo.py /path/to/volume.zarr [--port 5000] [--host 127.0.0.1] [--name mylayer]
#
# This starts a CORS-enabled local HTTP server for the directory that contains your .zarr,
# then opens a Neuroglancer Demo URL pointing at:  zarr://http://HOST:PORT/<basename>.zarr/
#
# Requirements: Python 3.8+, no extra pip packages needed.

import argparse, json, os, threading, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote
import tempfile

NEUROGLANCER_DEMO = "https://neuroglancer-demo.appspot.com/#!"

running_servers = []

class CORSRequestHandler(SimpleHTTPRequestHandler):
    # Serve files from a specific directory (passed via constructor)
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)
    def end_headers(self):
        # Allow cross-origin requests from any origin (needed for demo site)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.end_headers()

def is_port_in_use(host, port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    
def start_server(root_dir, host, port):
    handler = lambda *a, **kw: CORSRequestHandler(*a, directory=root_dir, **kw)
    httpd = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    running_servers.append(httpd)
    return httpd

def stop_all_servers():
    for s in running_servers:
        s.shutdown()
        s.server_close()
    running_servers.clear()

def build_state(file_http_url, layer_name, file_type='zarr'):
    """
    Build a Neuroglancer state with one or more layers.
    Neuroglancer infers shape/chunks from Zarr metadata; voxel size defaults to 1x1x1.

    Supports either single values or lists for each argument:
      - file_http_url: str or list of str
      - layer_name: str or list of str
      - file_type: str or list of str (defaults to 'zarr' for all if single string)

    Returns:
        dict: A Neuroglancer state dictionary.
    """
    if not isinstance(file_http_url, list):
        file_http_url = [file_http_url]
    if not isinstance(layer_name, list):
        layer_name = [layer_name]
    if not isinstance(file_type, list):
        file_type = [file_type]

    if len(file_type) < len(file_http_url):
        if len(file_type) == 1:
            file_type = file_type * len(file_http_url)
        else:
            raise SystemError(f"Unexpected number of file_types are defined: {file_type}")

    layers = [
        {
            "type": "image",
            "source": f"{ftype}://{url}",
            "name": name
        }
        for url, name, ftype in zip(file_http_url, layer_name, file_type)
    ]
    # You can add "crossSectionScale", "position", "layout", etc., if desired.

    return {"layers": layers}

def main():
    ap = argparse.ArgumentParser(description="Open a local Zarr store in Neuroglancer Demo.")
    ap.add_argument("file_path", nargs='+', help="Path to your .zarr directories")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind local HTTP server (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Port for local HTTP server (default: 5000)")
    ap.add_argument("--name", nargs='*', default=None, help="Layer names (default: basename of input file)")
    ap.add_argument("--file_type", default=None, help="file type (default: zarr)")
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open the browser")
    args = ap.parse_args()

    file_paths = [os.path.abspath(fp.rstrip("/")) for fp in args.file_path]
    for fp in file_paths:
        if args.file_type is None or args.file_type == 'zarr':
            if not os.path.isdir(fp) or not fp.endswith(".zarr"):
                raise SystemExit(f"Please provide paths to Zarr directories ending in .zarr: {fp}")
        elif args.file_type == 'nii.gz':
            if os.path.isdir(fp) or not fp.endswith(".nii.gz"):
                raise SystemExit(f"Please provide paths to NIfTI files ending in .nii.gz: {fp}")
            

    # Create symlinks of all input files into default root_dir
    root_dir = tempfile.mkdtemp(prefix="zarr_server_")
    for fp in file_paths:
        base = os.path.basename(fp.rstrip("/"))
        link_path = os.path.join(root_dir, base)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(fp, link_path)

    if args.name and len(args.name) != len(file_paths):
        raise SystemExit("Number of --name arguments must match number of input files.")
    layer_names = args.name or [os.path.basename(fp) for fp in file_paths]

    # 1) Start a CORS-enabled server for the parent directory
    if not is_port_in_use(args.host, args.port):
        httpd = start_server(root_dir, args.host, args.port)
    else:
        raise SystemExit(f"Port '{args.port}' already in use; please select a different one.")

    # 2) Build Neuroglancer state pointing to the served Zarr
    file_http_urls = [f"http://{args.host}:{args.port}/{os.path.basename(fp)}/" for fp in file_paths]
    if args.file_type is not None:
        state = build_state(file_http_urls, layer_names, file_type=args.file_type)
    else:
        state = build_state(file_http_urls, layer_names)

    # 3) Encode the state and construct the demo URL
    state_json = json.dumps(state, separators=(",", ":"))
    encoded = quote(state_json, safe="")
    url = NEUROGLANCER_DEMO + encoded

    print("\nNeuroglancer Demo URL (encoded):")
    print(url, "\n")

    # 4) Open in the default browser
    if not args.no_open:
        webbrowser.open(url)

    # Keep the server alive until Ctrl+C
    try:
        threading.Event().wait()  # sleep forever in main thread
    except KeyboardInterrupt:
        stop_all_servers()
        pass

if __name__ == "__main__":
    main()