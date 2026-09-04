"""Pretty-print a traced program (the IR tree) -- the `--dump` view.

Useful for checking what the tracer actually captured before asking why the
picture looks wrong, and as the stand-in for `qm.generate_qua_script`.
"""

from . import ir


def program_to_text(program, indent=0):
    out = []
    if getattr(program, "variables", None):
        for v in program.variables:
            init = "" if v.init is None else f" = {v.init}"
            size = f"[{v.size}]" if v.size else ""
            out.append(f"  declare {v.type} {v.name}{size}{init}")
    out.append("")
    out.extend(_body(program.body, 1))
    return "\n".join(out)


def _body(body, depth):
    pad = "  " * depth
    out = []
    for s in body:
        out.extend(_stmt(s, pad, depth))
    return out


def _stmt(s, pad, depth):
    loc = f"    # {s.loc}" if getattr(s, "loc", None) else ""
    t = type(s)
    if t is ir.Play:
        d = f", duration={_e(s.duration)}" if s.duration is not None else ""
        c = f", condition={_e(s.condition)}" if s.condition is not None else ""
        return [f"{pad}play({s.pulse!r}, {s.element!r}{d}{c}){loc}"]
    if t is ir.Measure:
        d = (", " + ", ".join(s.demods)) if s.demods else ""
        return [f"{pad}measure({s.pulse!r}, {s.element!r}{d}){loc}"]
    if t is ir.Wait:
        return [f"{pad}wait({_e(s.duration)}, {', '.join(map(repr, s.elements))}){loc}"]
    if t is ir.Align:
        return [f"{pad}align({', '.join(map(repr, s.elements))}){loc}"]
    if t is ir.Assign:
        return [f"{pad}assign({_e(s.target)}, {_e(s.value)}){loc}"]
    if t is ir.Save:
        return [f"{pad}save({_e(s.source)}, {s.stream!r}){loc}"]
    if t is ir.For:
        head = (f"{pad}for_({_e(s.var)}, {_e(s.init)}, {_e(s.cond)}, "
                f"{_e(s.update)}):{loc}")
        return [head] + _body(s.body, depth + 1)
    if t is ir.ForEach:
        vals = ", ".join(_vals(v) for v in s.values)
        return [f"{pad}for_each_(({', '.join(_e(v) for v in s.vars)}), "
                f"({vals})):{loc}"] + _body(s.body, depth + 1)
    if t is ir.While:
        return [f"{pad}while_({_e(s.cond)}):{loc}"] + _body(s.body, depth + 1)
    if t is ir.InfiniteLoop:
        return [f"{pad}infinite_loop_():{loc}"] + _body(s.body, depth + 1)
    if t is ir.If:
        out = [f"{pad}if_({_e(s.cond)}):{loc}"] + _body(s.body, depth + 1)
        for c, b in s.elifs:
            out += [f"{pad}elif_({_e(c)}):"] + _body(b, depth + 1)
        if s.orelse is not None:
            out += [f"{pad}else_():"] + _body(s.orelse, depth + 1)
        return out
    if t is ir.Switch:
        out = [f"{pad}switch_({_e(s.expr)}):{loc}"]
        for v, b in s.cases:
            out += [f"{pad}  case_({_e(v)}):"] + _body(b, depth + 2)
        if s.default is not None:
            out += [f"{pad}  default_():"] + _body(s.default, depth + 2)
        return out
    if t is ir.UpdateFrequency:
        units = "" if s.units == "Hz" else f", {s.units!r}"
        return [f"{pad}update_frequency({s.element!r}, {_e(s.frequency)}"
                f"{units}){loc}"]
    if t is ir.FrameRotation:
        return [f"{pad}frame_rotation({_e(s.angle)}, "
                f"{', '.join(map(repr, s.elements))}){loc}"]
    if t is ir.ResetFrame:
        return [f"{pad}reset_frame({', '.join(map(repr, s.elements))}){loc}"]
    if t is ir.ResetPhase:
        return [f"{pad}reset_phase({s.element!r}){loc}"]
    if t is ir.SetDCOffset:
        return [f"{pad}set_dc_offset({s.element!r}, {s.element_input!r}, "
                f"{_e(s.offset)}){loc}"]
    if t is ir.RampToZero:
        return [f"{pad}ramp_to_zero({s.element!r}, {_e(s.duration)}){loc}"]
    if t is ir.WaitForTrigger:
        return [f"{pad}wait_for_trigger({s.element!r}){loc}"]
    if t is ir.Pause:
        return [f"{pad}pause(){loc}"]
    if t is ir.StrictTiming:
        return [f"{pad}strict_timing_():{loc}"] + _body(s.body, depth + 1)
    if t is ir.Unknown:
        return [f"{pad}{s.name}({s.detail})   # not modelled{loc}"]
    fields = ", ".join(f"{k}={v!r}" for k, v in vars(s).items() if k != "loc")
    return [f"{pad}{t.__name__.lower()}({fields}){loc}"]


def _e(x):
    return repr(x) if x is not None else "None"


def _vals(v, n=6):
    body = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v[:n])
    return f"[{body}{', ...' if len(v) > n else ''}]"
