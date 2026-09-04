"""IR for a traced QUA program.

Two layers:

*Expressions* -- QUA is an embedded DSL, so `n < 100` inside `for_(...)` cannot
be evaluated at trace time (the variable has no value yet). Operator overloading
on `QuaVar` therefore builds an expression tree (`BinOp`, `Call`, ...) which the
interpreter evaluates later, once per unrolled iteration, against a concrete
environment.

*Statements* -- every QUA statement is recorded as a node. Blocks (`for_`,
`if_`, ...) hold child statement lists, so the traced program is a tree that
mirrors the source, NOT a flat schedule. Turning it into a schedule is the
interpreter's job (interp.py), because that requires unrolling.

Every node carries `loc`, the (file, line) of the user's source that produced
it, so the diagram can label segments with where they came from.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ------------------------------------------------------------------ source loc

@dataclass(frozen=True)
class Loc:
    """Where in the user's script a statement came from."""
    file: str = "?"
    line: int = 0

    def __str__(self):
        import os
        return f"{os.path.basename(self.file)}:{self.line}" if self.line else "?"


# ------------------------------------------------------------------ expressions

class Expr:
    """Base for QUA expression trees. Operators build nodes, never values."""

    # arithmetic ---------------------------------------------------------
    def __add__(self, o): return BinOp("+", self, lit(o))
    def __radd__(self, o): return BinOp("+", lit(o), self)
    def __sub__(self, o): return BinOp("-", self, lit(o))
    def __rsub__(self, o): return BinOp("-", lit(o), self)
    def __mul__(self, o): return BinOp("*", self, lit(o))
    def __rmul__(self, o): return BinOp("*", lit(o), self)
    def __truediv__(self, o): return BinOp("/", self, lit(o))
    def __rtruediv__(self, o): return BinOp("/", lit(o), self)
    def __floordiv__(self, o): return BinOp("//", self, lit(o))
    def __mod__(self, o): return BinOp("%", self, lit(o))
    def __neg__(self): return UnOp("-", self)
    def __lshift__(self, o): return BinOp("<<", self, lit(o))
    def __rshift__(self, o): return BinOp(">>", self, lit(o))
    def __xor__(self, o): return BinOp("^", self, lit(o))

    # comparison / logic -------------------------------------------------
    def __lt__(self, o): return BinOp("<", self, lit(o))
    def __le__(self, o): return BinOp("<=", self, lit(o))
    def __gt__(self, o): return BinOp(">", self, lit(o))
    def __ge__(self, o): return BinOp(">=", self, lit(o))
    def __eq__(self, o): return BinOp("==", self, lit(o))
    def __ne__(self, o): return BinOp("!=", self, lit(o))
    def __and__(self, o): return BinOp("&", self, lit(o))
    def __rand__(self, o): return BinOp("&", lit(o), self)
    def __or__(self, o): return BinOp("|", self, lit(o))
    def __ror__(self, o): return BinOp("|", lit(o), self)
    def __invert__(self): return UnOp("~", self)

    __hash__ = object.__hash__

    def __bool__(self):
        raise TypeError(
            "a QUA expression has no value at trace time -- use `with if_(cond):` "
            "instead of `if cond:` (and `&`/`|` instead of `and`/`or`)"
        )


@dataclass(eq=False)
class Lit(Expr):
    value: Any
    def __repr__(self): return repr(self.value)


@dataclass(eq=False)
class QuaVar(Expr):
    """A `declare()`d variable. Arrays carry `size`; scalars have size None."""
    name: str
    type: str = "int"          # 'int' | 'fixed' | 'bool'
    size: Optional[int] = None
    init: Any = None
    is_input_stream: bool = False

    def __getitem__(self, idx): return Index(self, lit(idx))
    def length(self): return Call("array_length", [self])
    def __repr__(self): return self.name


@dataclass(eq=False)
class Index(Expr):
    base: QuaVar
    index: Expr
    def __repr__(self): return f"{self.base!r}[{self.index!r}]"


@dataclass(eq=False)
class BinOp(Expr):
    op: str
    lhs: Expr
    rhs: Expr
    def __repr__(self): return f"({self.lhs!r} {self.op} {self.rhs!r})"


@dataclass(eq=False)
class UnOp(Expr):
    op: str
    operand: Expr
    def __repr__(self): return f"({self.op}{self.operand!r})"


@dataclass(eq=False)
class Call(Expr):
    """A library call: `Math.cos`, `Cast.to_int`, `Random.rand_int`, ..."""
    fn: str
    args: list
    def __repr__(self): return f"{self.fn}({', '.join(map(repr, self.args))})"


def lit(v):
    """Wrap a plain Python value as an expression (pass expressions through)."""
    return v if isinstance(v, Expr) else Lit(v)


# ------------------------------------------------------------------ pulse specs

@dataclass
class AmpSpec:
    """`amp(v)` or `amp(v00, v01, v10, v11)`; entries may be expressions."""
    values: list          # 1 or 4 entries

    @property
    def scalar(self): return self.values[0] if len(self.values) == 1 else None


@dataclass
class PulseRef:
    """What `play()` was asked to play: an operation name, optionally scaled,
    or a `ramp(rate)`."""
    operation: Optional[str] = None
    amp: Optional[AmpSpec] = None
    ramp_rate: Any = None      # set instead of `operation` for ramp()

    def __repr__(self):
        if self.ramp_rate is not None:
            return f"ramp({self.ramp_rate!r})"
        return self.operation + (f"*amp({self.amp.values[0]!r})" if self.amp else "")


# ------------------------------------------------------------------ statements

@dataclass
class Stmt:
    loc: Loc = field(default_factory=Loc, kw_only=True)


@dataclass
class Play(Stmt):
    pulse: PulseRef
    element: str
    duration: Any = None        # clock cycles (expr or int), None = pulse length
    condition: Any = None
    truncate: Any = None
    chirp: Any = None
    timestamp_stream: Any = None


@dataclass
class Measure(Stmt):
    pulse: PulseRef
    element: str
    stream: Any = None
    demods: list = field(default_factory=list)   # descriptive strings
    targets: list = field(default_factory=list)  # QuaVars written by demods
    duration: Any = None


@dataclass
class Wait(Stmt):
    duration: Any                # clock cycles
    elements: list = field(default_factory=list)


@dataclass
class Align(Stmt):
    elements: list = field(default_factory=list)  # empty = all used elements


@dataclass
class Assign(Stmt):
    target: Any
    value: Any


@dataclass
class Save(Stmt):
    source: Any
    stream: Any


@dataclass
class Pause(Stmt):
    pass


@dataclass
class UpdateFrequency(Stmt):
    element: str
    frequency: Any
    units: str = "Hz"
    keep_phase: bool = False


@dataclass
class FrameRotation(Stmt):
    angle: Any                   # radians
    elements: list = field(default_factory=list)


@dataclass
class ResetFrame(Stmt):
    elements: list = field(default_factory=list)


@dataclass
class ResetPhase(Stmt):
    element: str


@dataclass
class SetDCOffset(Stmt):
    element: str
    element_input: str
    offset: Any


@dataclass
class RampToZero(Stmt):
    element: str
    duration: Any = None         # clock cycles


@dataclass
class WaitForTrigger(Stmt):
    element: str
    pulse: Optional[PulseRef] = None


@dataclass
class UpdateCorrection(Stmt):
    element: str
    values: list = field(default_factory=list)


@dataclass
class Unknown(Stmt):
    """A QUA statement we record but do not model (no timing effect)."""
    name: str
    detail: str = ""


# --- blocks ---------------------------------------------------------------

@dataclass
class For(Stmt):
    var: Any
    init: Any
    cond: Any
    update: Any
    body: list = field(default_factory=list)


@dataclass
class ForEach(Stmt):
    vars: list
    values: list                 # list of concrete value-lists (one per var)
    body: list = field(default_factory=list)


@dataclass
class While(Stmt):
    cond: Any
    body: list = field(default_factory=list)


@dataclass
class InfiniteLoop(Stmt):
    body: list = field(default_factory=list)


@dataclass
class If(Stmt):
    cond: Any
    body: list = field(default_factory=list)
    elifs: list = field(default_factory=list)   # [(cond, body), ...]
    orelse: Optional[list] = None
    unsafe: bool = False                        # if_(..., unsafe=True)


@dataclass
class Switch(Stmt):
    expr: Any
    cases: list = field(default_factory=list)   # [(value, body), ...]
    default: Optional[list] = None


@dataclass
class StrictTiming(Stmt):
    body: list = field(default_factory=list)


@dataclass
class Program:
    """A traced `with program() as p:` block."""
    body: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    streams: list = field(default_factory=list)
    stream_processing: list = field(default_factory=list)
    source: str = "?"
