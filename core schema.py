"""
SYMBEX v2 — Core Schema
========================
Same public contract as v1 (NodeType, EdgeType, GraphNode, GraphEdge, TaskGraph,
TransformRecord, BenchmarkItem) so downstream notebook cells (agents, judge,
evaluation) do not need to change their interfaces — only the generators
producing BenchmarkItems change underneath.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class NodeType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    RESOURCE = "resource"
    STATE = "state"
    ROLE = "role"
    POLICY = "policy"
    OUTPUT = "output"
    DATA_ASSET = "data_asset"


class EdgeType(str, Enum):
    REQUIRES = "requires"
    PRODUCES = "produces"
    DEPENDS_ON = "depends_on"
    CAN_ACCESS = "can_access"
    BLOCKED = "blocked"
    GOVERNED_BY = "governed_by"
    TRANSITIONS_TO = "transitions_to"
    INFORMS = "informs"
    OWNED_BY = "owned_by"


class TransformType(str, Enum):
    SP = "symmetry_preserving"
    SB = "symmetry_breaking"


@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    edge_type: EdgeType
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]

    def adjacency_text(self) -> str:
        lines = ["NODES:"]
        for n in self.nodes:
            a = ", ".join(f"{k}={v}" for k, v in n.attrs.items())
            lines.append(f"  {n.id} [{n.node_type.value}]" + (f" ({a})" if a else ""))
        lines.append("EDGES:")
        for e in self.edges:
            a = ", ".join(f"{k}={v}" for k, v in e.attrs.items())
            lines.append(f"  {e.src} --[{e.edge_type.value}]--> {e.dst}" + (f" ({a})" if a else ""))
        return "\n".join(lines)

    def to_nx(self):
        import networkx as nx
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(n.id, node_type=n.node_type, **n.attrs)
        for e in self.edges:
            G.add_edge(e.src, e.dst, edge_type=e.edge_type, **e.attrs)
        return G


@dataclass
class TransformRecord:
    transform_type: TransformType
    description: str
    expected_behavior: str


@dataclass
class ActionSpec:
    """A single canonical action, separated into name + ordered argument slots.
    This is what fixes the substring-matching grading problem: grading compares
    parsed (name, args) tuples, not raw strings.
    """
    name: str
    args: List[str] = field(default_factory=list)

    def render(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    @staticmethod
    def parse(raw: str) -> "ActionSpec":
        raw = raw.strip()
        if "(" not in raw:
            return ActionSpec(name=raw, args=[])
        name, rest = raw.split("(", 1)
        rest = rest.rstrip(")")
        args = [a.strip() for a in rest.split(",") if a.strip()] if rest else []
        return ActionSpec(name=name.strip(), args=args)

    def matches(self, other: "ActionSpec") -> bool:
        """Structural equality: same action name AND same argument values,
        order-independent on args (covers e.g. allocate(P1,50) vs allocate(amount=50,project=P1)
        style variation) but exact on the name.
        """
        if self.name != other.name:
            return False
        return sorted(self.args) == sorted(other.args)


@dataclass
class BenchmarkItem:
    task_id: str
    family: str
    template: str
    seed: int
    variant_label: str
    graph: TaskGraph
    goal_text: str
    constraint_text: str
    action_space: List[str]
    transform_record: Optional[TransformRecord]
    correct_action_sequence: List[str]
    forbidden_actions: List[str]
    difficulty: str
    # ── NEW in v2: sampled parameters used to generate this instance, kept for
    # transparency / reproducibility audits and for stratified analysis.
    sampled_params: Dict[str, Any] = field(default_factory=dict)
    # ── NEW in v2: explicit ordering sensitivity flag — some correct sequences
    # are strictly ordered (state machines), others are a set (independent
    # allocations); grading must respect this distinction.
    sequence_is_ordered: bool = True
