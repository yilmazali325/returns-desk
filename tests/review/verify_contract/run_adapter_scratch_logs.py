"""Run desk.adapter_main.main with server logs redirected to a scratch dir (argv[1]); rest = adapter args."""
import sys
sys.path.insert(0, "/Users/ali/Desktop/Projects/retail-returns-desk")
from pathlib import Path
import desk.adapter_main as am

log_dir = Path(sys.argv[1])
orig = am.ServerHandle.__init__
def init(self, profile, port, llama_url, log_dir_=None):
    orig(self, profile, port, llama_url, log_dir)
am.ServerHandle.__init__ = init
sys.exit(am.main(sys.argv[2:]))
