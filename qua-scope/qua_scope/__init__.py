"""qua-scope: interpret a QUA script and draw the pulse sequence it plays.

    from qua_scope import scope
    tl = scope("power_rabi.py", out="rabi.svg")

Pipeline: loader.load_script (run the script against a tracing stand-in for
`qm.qua`) -> ir.Program -> interp.Interpreter (unroll + schedule) -> Timeline ->
render_svg / render_text.
"""

from .interp import Interpreter, RunConfig, interpret
from .ir import Program
from .loader import load_script
from .machine import Machine
from .render_svg import render_svg
from .render_text import render_text
from .textdump import program_to_text
from .timeline import Timeline

__version__ = "0.1.0"
__all__ = ["scope", "load_script", "Interpreter", "RunConfig", "interpret",
           "Machine", "Timeline", "Program", "render_svg", "render_text",
           "program_to_text"]


def scope(path, out=None, iterations=2, modulate=False, config=None, **kw):
    """Load a QUA script, schedule it, and optionally write an SVG.

    Returns the Timeline so you can inspect segments programmatically.
    """
    res = load_script(path, extra_config=config)
    tl = Interpreter(Machine(res.config),
                     RunConfig(max_iterations=iterations, **kw)).run(res.program)
    if out:
        with open(out, "w") as f:
            f.write(render_svg(tl, modulate=modulate, title=path))
    return tl
