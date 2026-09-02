from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyzx as zx
from pyzx.extract import extract_circuit
from pyzx.local_search.congruences import is_lc_vertex, lc_cong
from pyzx.simplify import euler_expansion_rewrite, full_reduce, spider_simp, to_gh
from pyzx.utils import EdgeType, VertexType, toggle_edge
from qiskit import QuantumCircuit, qasm2, transpile

try:
    from mqt.bench import BenchmarkLevel, get_benchmark
except ImportError:  # Random-circuit mode still works without MQT Bench.
    BenchmarkLevel = None
    get_benchmark = None


# ============================================================
# Stable per-spider metadata keys
# ============================================================

# PyZX Graph.copy() can reindex raw graph vertex IDs. These values live in
# PyZX vdata, which PyZX copies together with the vertex, so they remain stable.
UID_KEY = "dqc_uid"
QPU_KEY = "dqc_qpu_owner"
LOGICAL_KEY = "dqc_logical_qubit"
ORIGIN_KEY = "dqc_origin"
CARRIER_KEY = "dqc_phase_carrier"
ANCHOR_UID_KEY = "dqc_phase_anchor_uid"


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class QPUConfig:
    """Static logical-qubit capacity of one QPU."""

    name: str
    logical_qubits: int


@dataclass(frozen=True)
class ExperimentConfig:
    # Logical circuit sizes to test.
    qubit_sizes: tuple[int, ...] = (4, 8, 12)

    # Random-circuit source.
    run_random_circuits: bool = True
    circuits_per_size: int = 10
    circuit_layers: int = 10
    single_qubit_gate_probability: float = 0.80
    cx_probability: float = 0.80

    # MQT Bench source. Each named benchmark is generated once per requested
    # size by default. Increase mqt_repetitions for repeated randomized
    # parameter instances of variational benchmarks.
    run_mqt_benchmarks: bool = True
    mqt_benchmarks: tuple[str, ...] = (
        "qft",
        "qftentangled",
        "graphstate",
        "qaoa",
        "grover",
    )
    mqt_repetitions: int = 1
    mqt_random_parameters: bool = True

    # Number of QPUs. If capacities are omitted, logical qubits are distributed
    # as evenly as possible.
    num_qpus: int = 2
    qpu_capacities_by_size: dict[int, tuple[int, ...]] | None = None

    # Warm start = best fixed logical-qubit placement among random samples.
    random_partitions: int = 10

    # Natural termination is "no strict improvement in a complete LC sweep".
    # This is only a defensive cap.
    safety_max_lc_steps: int = 1000

    # All random streams derive from this seed.
    master_seed: int = 20260901

    # Useful while developing; can be slow for a full benchmark sweep.
    verify_equivalence: bool = False

    # If one MQT benchmark/size pair cannot be generated or is non-unitary,
    # report and continue rather than losing the entire sweep.
    skip_failed_mqt: bool = True

    output_dir: str = "zx_lc_random_mqt_results"


# ============================================================
# Internal representations
# ============================================================


@dataclass
class ZXState:
    """Red/green ZX representation used for hypergraph scoring.

    Each internal spider carries a stable UID and immutable QPU owner in vdata.
    ``next_uid`` is used only when a rewrite introduces a genuinely new
    auxiliary spider. Auxiliary spiders do not consume another logical-qubit
    slot; they inherit a local QPU owner from the rewrite that created them.
    """

    graph: Any
    next_uid: int


@dataclass
class HypergraphEncoding:
    """X-spider vertices and phase-free Z-spider hyperedges."""

    state: ZXState
    x_uids: tuple[int, ...]
    edge_members: dict[int, frozenset[int]]  # z_uid -> X-spider UIDs
    edge_leg_count: dict[int, int]           # z_uid -> ZX degree of Z spider
    edge_anchor_owner: dict[int, int]
    node_incidence: dict[int, tuple[int, ...]]
    owner_by_uid: dict[int, int]


@dataclass(frozen=True)
class CutScore:
    cut_hyperedges: tuple[int, ...]       # stable Z-spider UIDs
    boundary_x_vertices: tuple[int, ...] # stable X-spider UIDs
    cut_leg_count: int                    # diagnostic only

    @property
    def cost(self) -> int:
        """Figure of merit: number of cut Z-hyperedges."""
        return len(self.cut_hyperedges)

    @property
    def objective(self) -> int:
        """The optimization objective is only the number of cut hyperedges."""
        return self.cost


@dataclass(frozen=True)
class WarmStart:
    logical_to_qpu: dict[int, int]
    score: CutScore
    sampled_objectives: tuple[int, ...]


@dataclass(frozen=True)
class LCRecord:
    step: int
    center_uid: int
    center_qpu: int
    neighbor_uids_before: tuple[int, ...]
    cuts_before: int
    cuts_after: int
    cut_legs_before: int
    cut_legs_after: int
    theoretical_clifford_count: int
    center_clifford: str
    neighbor_clifford: str
    phase_changes: tuple[tuple[int, str, str], ...]  # uid, before, after
    new_vertices: tuple[tuple[int, int, str], ...]   # uid, qpu, phase


@dataclass
class TrialResult:
    source_kind: str
    benchmark: str
    requested_num_qubits: int
    num_qubits: int
    trial: int

    circuit_seed: int
    transpiler_seed: int
    partition_seed: int

    baseline_cut_hyperedges: int
    optimized_cut_hyperedges: int
    baseline_cut_z_legs: int
    optimized_cut_z_legs: int
    accepted_lcs: int
    induced_cliffords: int

    equivalence_verified: bool | None

    @property
    def source_label(self) -> str:
        if self.source_kind == "random":
            return "random"
        return f"mqt:{self.benchmark}"

    @property
    def cut_reduction_fraction(self) -> float:
        if self.baseline_cut_hyperedges == 0:
            return 0.0
        return (
            self.baseline_cut_hyperedges - self.optimized_cut_hyperedges
        ) / self.baseline_cut_hyperedges


# ============================================================
# Small metadata helpers
# ============================================================


def _vdata(graph, vertex: int, key: str, default: Any = None) -> Any:
    return graph.vdata(vertex, key, default)


def _uid(graph, vertex: int) -> int:
    value = _vdata(graph, vertex, UID_KEY)
    if value is None:
        raise ValueError(f"Vertex {vertex} has no stable UID")
    return int(value)


def _owner(graph, vertex: int) -> int:
    value = _vdata(graph, vertex, QPU_KEY)
    if value is None:
        raise ValueError(f"Vertex UID {_uid(graph, vertex)} has no QPU owner")
    return int(value)


def _uid_index(graph) -> dict[int, int]:
    index: dict[int, int] = {}
    for vertex in graph.vertices():
        if graph.type(vertex) == VertexType.BOUNDARY:
            continue
        uid = _uid(graph, vertex)
        if uid in index:
            raise ValueError(f"Duplicate stable UID {uid}")
        index[uid] = vertex
    return index


def _phase_by_uid(
    graph,
    uids: Iterable[int],
    index: dict[int, int] | None = None,
) -> dict[int, str]:
    if index is None:
        index = _uid_index(graph)
    return {
        uid: str(graph.phase(index[uid]))
        for uid in uids
        if uid in index
    }


# ============================================================
# Reproducible random streams
# ============================================================


def derived_seed(master_seed: int, num_qubits: int, trial: int, stream: int) -> int:
    """Derive an independent deterministic 32-bit seed."""

    sequence = np.random.SeedSequence([master_seed, num_qubits, trial, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_process_globals(seed: int) -> None:
    """Defensive seeding for dependencies that use module-global RNGs."""

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Circuit sources and RX/RZ/CX compilation
# ============================================================


def random_logical_circuit(
    num_qubits: int,
    layers: int,
    seed: int,
    single_qubit_gate_probability: float,
    cx_probability: float,
) -> QuantumCircuit:
    """Generate a repeatable random unitary source circuit.

    H and RY are intentionally included so recompilation into {RX, RZ, CX}
    remains nontrivial.
    """

    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2")
    if layers < 1:
        raise ValueError("layers must be positive")

    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits, name=f"random_{num_qubits}q_{seed}")

    for _ in range(layers):
        for q in range(num_qubits):
            if rng.random() >= single_qubit_gate_probability:
                continue

            choice = int(rng.integers(0, 4))
            theta = float(rng.uniform(-np.pi, np.pi))
            if choice == 0:
                circuit.h(q)
            elif choice == 1:
                circuit.ry(theta, q)
            elif choice == 2:
                circuit.rz(theta, q)
            else:
                circuit.rx(theta, q)

        qubits = rng.permutation(num_qubits)
        for i in range(0, num_qubits - 1, 2):
            if rng.random() >= cx_probability:
                continue
            a = int(qubits[i])
            b = int(qubits[i + 1])
            if rng.random() < 0.5:
                circuit.cx(a, b)
            else:
                circuit.cx(b, a)

    return circuit


def mqt_benchmark_circuit(
    benchmark: str,
    circuit_size: int,
    seed: int,
    random_parameters: bool = True,
) -> QuantumCircuit:
    """Return an algorithm-level MQT Bench circuit as a Qiskit circuit."""

    if get_benchmark is None or BenchmarkLevel is None:
        raise ImportError(
            "MQT Bench is not installed. Install it with:\n"
            "    pip install mqt-bench\n"
            "Random-circuit mode can still run without it."
        )

    # MQT Bench may use global RNGs for randomized parameters.
    seed_process_globals(seed)

    circuit = get_benchmark(
        benchmark=benchmark,
        level=BenchmarkLevel.ALG,
        circuit_size=circuit_size,
        random_parameters=random_parameters,
    )

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"MQT Bench returned {type(circuit)!r}, expected QuantumCircuit"
        )

    # The LC pipeline models a unitary logical computation. Strip terminal
    # readout measurements; reject genuinely dynamic/non-unitary benchmarks.
    circuit = circuit.remove_final_measurements(inplace=False)
    if circuit.num_clbits:
        raise ValueError(
            f"MQT benchmark {benchmark!r} still contains classical bits after "
            "removing final measurements; dynamic circuits are not supported."
        )

    return circuit


def make_source_circuit(
    source_kind: str,
    requested_num_qubits: int,
    trial: int,
    config: ExperimentConfig,
    circuit_seed: int,
    benchmark: str | None = None,
) -> QuantumCircuit:
    """Create one logical source circuit from either supported source."""

    if source_kind == "random":
        return random_logical_circuit(
            num_qubits=requested_num_qubits,
            layers=config.circuit_layers,
            seed=circuit_seed,
            single_qubit_gate_probability=config.single_qubit_gate_probability,
            cx_probability=config.cx_probability,
        )

    if source_kind == "mqt":
        if benchmark is None:
            raise ValueError("benchmark must be provided for source_kind='mqt'")
        return mqt_benchmark_circuit(
            benchmark=benchmark,
            circuit_size=requested_num_qubits,
            seed=circuit_seed,
            random_parameters=config.mqt_random_parameters,
        )

    raise ValueError(f"Unknown circuit source {source_kind!r}")


def compile_to_rx_rz_cx(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    """Compile every source through the same universal basis {RX, RZ, CX}."""

    compiled = transpile(
        circuit,
        basis_gates=["rx", "rz", "cx"],
        optimization_level=1,
        seed_transpiler=seed,
    )

    unexpected = set(compiled.count_ops()) - {"rx", "rz", "cx", "barrier"}
    if unexpected:
        raise RuntimeError(f"Unexpected gates after basis compilation: {unexpected}")

    return compiled


def circuit_to_zx(circuit: QuantumCircuit):
    """Use PyZX's circuit ``to_graph()`` conversion directly."""

    pyzx_circuit = zx.Circuit.from_qasm(qasm2.dumps(circuit))
    return pyzx_circuit.to_graph(compress_rows=False)


# ============================================================
# Static QPU placement
# ============================================================


def equal_qpus(num_qubits: int, num_qpus: int) -> tuple[QPUConfig, ...]:
    if not 1 <= num_qpus <= num_qubits:
        raise ValueError("Require 1 <= num_qpus <= num_qubits")

    base, remainder = divmod(num_qubits, num_qpus)
    return tuple(
        QPUConfig(name=f"qpu_{i}", logical_qubits=base + int(i < remainder))
        for i in range(num_qpus)
    )


def qpus_for_size(num_qubits: int, config: ExperimentConfig) -> tuple[QPUConfig, ...]:
    if config.qpu_capacities_by_size is None:
        return equal_qpus(num_qubits, config.num_qpus)

    capacities = config.qpu_capacities_by_size.get(num_qubits)
    if capacities is None:
        raise ValueError(f"No QPU capacities configured for {num_qubits} qubits")
    if len(capacities) != config.num_qpus:
        raise ValueError(
            f"Expected {config.num_qpus} QPU capacities for {num_qubits}q; "
            f"received {len(capacities)}"
        )
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("QPU capacities must be positive")
    if sum(capacities) < num_qubits:
        raise ValueError(
            f"Total capacity {sum(capacities)} is less than circuit size {num_qubits}"
        )

    return tuple(
        QPUConfig(name=f"qpu_{i}", logical_qubits=int(capacity))
        for i, capacity in enumerate(capacities)
    )


def random_valid_logical_placement(
    num_qubits: int,
    qpus: Sequence[QPUConfig],
    rng: np.random.Generator,
) -> dict[int, int]:
    """Sample a capacity-valid static logical-qubit-to-QPU placement."""

    slots = [
        qpu
        for qpu, config in enumerate(qpus)
        for _ in range(config.logical_qubits)
    ]
    if len(slots) < num_qubits:
        raise ValueError("Insufficient total QPU capacity")

    rng.shuffle(slots)
    selected_slots = slots[:num_qubits]

    logical_qubits = list(range(num_qubits))
    rng.shuffle(logical_qubits)

    return {
        logical: int(qpu)
        for logical, qpu in zip(logical_qubits, selected_slots)
    }


# ============================================================
# Initial circuit-derived ZX state
# ============================================================


def _logical_qubit_from_coordinate(graph, vertex: int, num_qubits: int) -> int:
    coordinate = float(graph.qubit(vertex))
    logical = int(round(coordinate))
    if abs(coordinate - logical) > 1e-8 or not 0 <= logical < num_qubits:
        raise ValueError(
            f"Initial spider {vertex} has non-logical qubit coordinate {coordinate}. "
            "The fixed-placement model expects a circuit-derived RX/RZ/CX ZX "
            "diagram before LC."
        )
    return logical


def prepare_initial_zx(raw_graph, num_qubits: int) -> ZXState:
    """Fuse initial same-color worldline spiders and attach stable metadata."""

    graph = raw_graph.copy()
    spider_simp(graph)

    next_uid = 0
    for vertex in sorted(graph.vertices()):
        if graph.type(vertex) not in (VertexType.X, VertexType.Z):
            continue

        logical = _logical_qubit_from_coordinate(graph, vertex, num_qubits)
        graph.set_vdata(vertex, UID_KEY, next_uid)
        graph.set_vdata(vertex, LOGICAL_KEY, logical)
        graph.set_vdata(vertex, ORIGIN_KEY, "initial")
        next_uid += 1

    for edge in graph.edges():
        u, v = graph.edge_st(edge)
        if VertexType.BOUNDARY in (graph.type(u), graph.type(v)):
            continue
        if graph.edge_type(edge) != EdgeType.SIMPLE:
            raise ValueError("Initial RX/RZ/CX ZX diagram contains an internal H edge")
        if {graph.type(u), graph.type(v)} != {VertexType.X, VertexType.Z}:
            raise ValueError(
                f"Initial simplified ZX graph is not X/Z bipartite: {u}--{v}"
            )

    return ZXState(graph=graph, next_uid=next_uid)


def _assign_fixed_qpu_owners(state: ZXState, logical_to_qpu: dict[int, int]) -> ZXState:
    graph = state.graph.copy()

    for vertex in graph.vertices():
        if graph.type(vertex) not in (VertexType.X, VertexType.Z):
            continue
        logical = _vdata(graph, vertex, LOGICAL_KEY)
        if logical is None:
            raise ValueError(f"Initial spider UID {_uid(graph, vertex)} has no logical qubit")
        graph.set_vdata(vertex, QPU_KEY, int(logical_to_qpu[int(logical)]))

    return ZXState(graph=graph, next_uid=state.next_uid)


# ============================================================
# Z-phase normalization
# ============================================================


def _is_phase_carrier(graph, vertex: int) -> bool:
    return bool(_vdata(graph, vertex, CARRIER_KEY, False))


def _extract_z_phases_inplace(graph, next_uid: int) -> int:
    """Split phases from communication Z spiders in place."""

    z_vertices = [
        vertex
        for vertex in list(graph.vertices())
        if graph.type(vertex) == VertexType.Z and not _is_phase_carrier(graph, vertex)
    ]

    for z in z_vertices:
        phase = graph.phase(z)
        if phase == 0:
            continue

        x_neighbors = sorted(
            (n for n in graph.neighbors(z) if graph.type(n) == VertexType.X),
            key=lambda n: (graph.row(n), _uid(graph, n)),
        )
        if not x_neighbors:
            continue

        x = x_neighbors[0]
        edge = graph.edge(z, x)
        if graph.edge_type(edge) != EdgeType.SIMPLE:
            raise ValueError("Z-phase extraction requires a simple Z--X edge")

        carrier = graph.add_vertex(
            ty=VertexType.Z,
            qubit=0.5 * (graph.qubit(z) + graph.qubit(x)),
            row=0.5 * (graph.row(z) + graph.row(x)),
            phase=phase,
        )
        graph.set_vdata(carrier, UID_KEY, next_uid)
        graph.set_vdata(carrier, QPU_KEY, _owner(graph, z))
        graph.set_vdata(carrier, ORIGIN_KEY, "phase_carrier")
        graph.set_vdata(carrier, CARRIER_KEY, True)
        graph.set_vdata(carrier, ANCHOR_UID_KEY, _uid(graph, z))

        logical = _vdata(graph, z, LOGICAL_KEY)
        if logical is not None:
            graph.set_vdata(carrier, LOGICAL_KEY, int(logical))

        next_uid += 1
        graph.set_phase(z, 0)
        graph.remove_edge(edge)
        graph.add_edge((z, carrier), edgetype=EdgeType.SIMPLE)
        graph.add_edge((carrier, x), edgetype=EdgeType.SIMPLE)

    return next_uid


def _extract_z_phases(state: ZXState) -> ZXState:
    graph = state.graph.copy()
    next_uid = _extract_z_phases_inplace(graph, state.next_uid)
    normalized = ZXState(graph=graph, next_uid=next_uid)
    _validate_scoring_state(normalized)
    return normalized


def _phase_carriers(graph) -> list[int]:
    return sorted(
        (
            v
            for v in graph.vertices()
            if graph.type(v) == VertexType.Z and _is_phase_carrier(graph, v)
        ),
        key=lambda v: _uid(graph, v),
    )


def _fuse_phase_carriers(state: ZXState) -> ZXState:
    """Exact inverse of our phase splitting, preserving each anchor UID."""

    graph = state.graph.copy()
    index = _uid_index(graph)
    carrier_uids = [_uid(graph, v) for v in _phase_carriers(graph)]

    for carrier_uid in carrier_uids:
        carrier = index.pop(carrier_uid)
        anchor_uid = int(_vdata(graph, carrier, ANCHOR_UID_KEY))
        anchor = index[anchor_uid]

        neighbors = list(graph.neighbors(carrier))
        if len(neighbors) != 2 or anchor not in neighbors:
            raise ValueError("Malformed phase carrier")
        other = neighbors[0] if neighbors[1] == anchor else neighbors[1]

        if graph.edge_type(graph.edge(carrier, anchor)) != EdgeType.SIMPLE:
            raise ValueError("Carrier-anchor edge must be simple")
        if graph.edge_type(graph.edge(carrier, other)) != EdgeType.SIMPLE:
            raise ValueError("Carrier-other edge must be simple")
        if graph.connected(anchor, other):
            raise ValueError("Unexpected anchor-other edge while fusing phase carrier")

        graph.add_to_phase(anchor, graph.phase(carrier))
        graph.remove_vertex(carrier)
        graph.add_edge((anchor, other), edgetype=EdgeType.SIMPLE)

    return ZXState(graph=graph, next_uid=state.next_uid)


# ============================================================
# Hypergraph scoring representation
# ============================================================


def validate_hypergraph_ready(
    state: ZXState,
    uid_index: dict[int, int] | None = None,
) -> None:
    """Reject malformed ZX structure before hypergraph construction."""

    graph = state.graph
    if uid_index is None:
        uid_index = _uid_index(graph)

    carrier_pairs: set[frozenset[int]] = set()
    for carrier in _phase_carriers(graph):
        anchor_uid = int(_vdata(graph, carrier, ANCHOR_UID_KEY))
        if anchor_uid not in uid_index:
            raise ValueError("Phase carrier references a missing anchor")
        carrier_pairs.add(frozenset((_uid(graph, carrier), anchor_uid)))

    for edge in graph.edges():
        u, v = graph.edge_st(edge)
        tu, tv = graph.type(u), graph.type(v)

        if VertexType.BOUNDARY in (tu, tv):
            continue

        if graph.edge_type(edge) != EdgeType.SIMPLE:
            raise ValueError(
                "Hypergraph construction encountered a residual internal H edge"
            )

        if {tu, tv} == {VertexType.X, VertexType.Z}:
            continue

        uid_pair = frozenset((_uid(graph, u), _uid(graph, v)))
        if tu == tv == VertexType.Z and uid_pair in carrier_pairs:
            continue

        raise ValueError(
            "Hypergraph construction requires simple X--Z incidence; "
            f"unexpected internal edge {tuple(sorted(uid_pair))} "
            f"with endpoint types ({tu}, {tv})"
        )


def _validate_scoring_state(state: ZXState) -> dict[int, int]:
    graph = state.graph
    uid_index = _uid_index(graph)

    for uid, vertex in uid_index.items():
        if _vdata(graph, vertex, QPU_KEY) is None:
            raise ValueError(f"Spider UID {uid} is missing QPU placement metadata")

    for carrier in _phase_carriers(graph):
        anchor_uid = int(_vdata(graph, carrier, ANCHOR_UID_KEY))
        if anchor_uid not in uid_index:
            raise ValueError("Phase carrier references a missing anchor")
        anchor = uid_index[anchor_uid]

        if graph.vertex_degree(carrier) != 2:
            raise ValueError("Phase carrier must have degree 2")
        if graph.phase(anchor) != 0:
            raise ValueError("Communication hyperedge anchor must be phase-free")
        if _owner(graph, carrier) != _owner(graph, anchor):
            raise ValueError("Phase carrier must inherit anchor QPU metadata")

    validate_hypergraph_ready(state, uid_index=uid_index)
    return uid_index


def build_hypergraph(state: ZXState) -> HypergraphEncoding:
    """Build the X-vertex/Z-hyperedge abstraction used for every score."""

    graph = state.graph
    uid_index = _validate_scoring_state(state)
    owner_by_uid = {uid: _owner(graph, vertex) for uid, vertex in uid_index.items()}

    x_uids = tuple(
        sorted(
            _uid(graph, v)
            for v in graph.vertices()
            if graph.type(v) == VertexType.X
        )
    )
    x_uid_set = set(x_uids)

    carriers_by_anchor_uid: dict[int, list[int]] = {}
    carrier_uid_set: set[int] = set()
    for carrier in _phase_carriers(graph):
        carrier_uid = _uid(graph, carrier)
        anchor_uid = int(_vdata(graph, carrier, ANCHOR_UID_KEY))
        carrier_uid_set.add(carrier_uid)
        carriers_by_anchor_uid.setdefault(anchor_uid, []).append(carrier_uid)

    edge_members: dict[int, frozenset[int]] = {}
    edge_leg_count: dict[int, int] = {}
    edge_anchor_owner: dict[int, int] = {}
    incidence_lists: dict[int, list[int]] = {uid: [] for uid in x_uids}

    for z in graph.vertices():
        if graph.type(z) != VertexType.Z:
            continue
        z_uid = _uid(graph, z)
        if z_uid in carrier_uid_set:
            continue

        members = {
            _uid(graph, n)
            for n in graph.neighbors(z)
            if graph.type(n) == VertexType.X
        }

        for carrier_uid in carriers_by_anchor_uid.get(z_uid, ()):
            carrier = uid_index[carrier_uid]
            members.update(
                _uid(graph, n)
                for n in graph.neighbors(carrier)
                if graph.type(n) == VertexType.X
            )

        if not members:
            continue
        if graph.phase(z) != 0:
            raise ValueError(f"Communication Z spider UID {z_uid} is not phase-free")
        if not members <= x_uid_set:
            raise AssertionError("Hyperedge contains a non-X member")

        frozen = frozenset(members)
        edge_members[z_uid] = frozen
        edge_leg_count[z_uid] = graph.vertex_degree(z)
        edge_anchor_owner[z_uid] = _owner(graph, z)

        for x_uid in frozen:
            incidence_lists[x_uid].append(z_uid)

    return HypergraphEncoding(
        state=state,
        x_uids=x_uids,
        edge_members=edge_members,
        edge_leg_count=edge_leg_count,
        edge_anchor_owner=edge_anchor_owner,
        node_incidence={
            uid: tuple(sorted(edges)) for uid, edges in incidence_lists.items()
        },
        owner_by_uid=owner_by_uid,
    )


def score_hypergraph(encoding: HypergraphEncoding) -> CutScore:
    """Score the number of Z hyperedges whose support spans multiple QPUs."""

    cut_edges: list[int] = []
    boundary_x: set[int] = set()
    cut_leg_count = 0

    for z_uid, members in encoding.edge_members.items():
        touched = {encoding.owner_by_uid[x_uid] for x_uid in members}
        touched.add(encoding.edge_anchor_owner[z_uid])

        if len(touched) > 1:
            cut_edges.append(z_uid)
            boundary_x.update(members)
            cut_leg_count += encoding.edge_leg_count[z_uid]

    return CutScore(
        cut_hyperedges=tuple(sorted(cut_edges)),
        boundary_x_vertices=tuple(sorted(boundary_x)),
        cut_leg_count=cut_leg_count,
    )


# ============================================================
# Warm start: best of random fixed logical placements
# ============================================================


def state_for_logical_placement(
    prepared_state: ZXState,
    logical_to_qpu: dict[int, int],
) -> ZXState:
    state = _assign_fixed_qpu_owners(prepared_state, logical_to_qpu)
    return _extract_z_phases(state)


def choose_warm_start(
    prepared_state: ZXState,
    num_qubits: int,
    qpus: Sequence[QPUConfig],
    num_samples: int,
    seed: int,
) -> tuple[ZXState, WarmStart]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    if len(qpus) == 1:
        if qpus[0].logical_qubits < num_qubits:
            raise ValueError("Single QPU does not have enough logical-qubit capacity")
        placement = {logical: 0 for logical in range(num_qubits)}
        state = state_for_logical_placement(prepared_state, placement)
        score = score_hypergraph(build_hypergraph(state))
        if score.cost != 0:
            raise AssertionError("One-QPU placement produced a nonzero cut score")
        return state, WarmStart(
            logical_to_qpu=placement,
            score=score,
            sampled_objectives=tuple(0 for _ in range(num_samples)),
        )

    rng = np.random.default_rng(seed)
    best_state: ZXState | None = None
    best_placement: dict[int, int] | None = None
    best_score: CutScore | None = None
    sampled_objectives: list[int] = []

    for _ in range(num_samples):
        placement = random_valid_logical_placement(num_qubits, qpus, rng)
        state = state_for_logical_placement(prepared_state, placement)
        score = score_hypergraph(build_hypergraph(state))
        sampled_objectives.append(score.cost)

        placement_key = tuple(placement[q] for q in range(num_qubits))
        if best_score is None:
            choose = True
        elif score.cost < best_score.cost:
            choose = True
        elif score.cost > best_score.cost:
            choose = False
        else:
            assert best_placement is not None
            best_key = tuple(best_placement[q] for q in range(num_qubits))
            choose = placement_key < best_key

        if choose:
            best_state = state
            best_placement = dict(placement)
            best_score = score

    assert best_state is not None and best_placement is not None and best_score is not None

    return best_state, WarmStart(
        logical_to_qpu=best_placement,
        score=best_score,
        sampled_objectives=tuple(sampled_objectives),
    )


# ============================================================
# Exact reference-color restoration around LC
# ============================================================


def make_graph_like_for_lc(state: ZXState) -> tuple[ZXState, ZXState]:
    """Return the fused RG reference state and its all-green LC view."""

    reference_rg = _fuse_phase_carriers(state)
    graph = reference_rg.graph.copy()
    to_gh(graph)

    for vertex in graph.vertices():
        if graph.type(vertex) not in (VertexType.Z, VertexType.BOUNDARY):
            raise ValueError("to_gh left a non-Z internal spider")

    for edge in graph.edges():
        u, v = graph.edge_st(edge)
        if VertexType.BOUNDARY in (graph.type(u), graph.type(v)):
            continue
        if graph.edge_type(edge) != EdgeType.HADAMARD:
            raise ValueError("Graph-like conversion left a non-H internal edge")

    return reference_rg, ZXState(graph=graph, next_uid=reference_rg.next_uid)


def _restore_reference_rg_colors(
    graph,
    reference_rg: ZXState,
    center_uid: int,
    new_lc_uids: Sequence[int],
) -> None:
    """Restore exact pre-LC X/Z colors after applying LC in the GH view."""

    reference_index = _uid_index(reference_rg.graph)
    candidate_index = _uid_index(graph)

    if center_uid not in reference_index:
        raise ValueError(f"LC center UID {center_uid} missing from RG reference")

    for uid in sorted(reference_index):
        reference_vertex = reference_index[uid]
        candidate_vertex = candidate_index.get(uid)
        if candidate_vertex is None:
            raise ValueError(f"LC unexpectedly removed pre-existing spider UID {uid}")

        reference_type = reference_rg.graph.type(reference_vertex)
        if reference_type not in (VertexType.X, VertexType.Z):
            raise ValueError(f"Unsupported reference spider type {reference_type}")

        if reference_type == VertexType.X:
            graph.set_type(candidate_vertex, VertexType.X)
            for edge in list(graph.incident_edges(candidate_vertex)):
                graph.set_edge_type(edge, toggle_edge(graph.edge_type(edge)))
        else:
            graph.set_type(candidate_vertex, VertexType.Z)

    center_type = reference_rg.graph.type(reference_index[center_uid])
    gadget_type = VertexType.X if center_type == VertexType.Z else VertexType.Z

    for uid in sorted(new_lc_uids):
        vertex = candidate_index.get(uid)
        if vertex is None:
            raise ValueError(f"New LC gadget UID {uid} disappeared during restoration")

        if gadget_type == VertexType.X:
            graph.set_type(vertex, VertexType.X)
            for edge in list(graph.incident_edges(vertex)):
                graph.set_edge_type(edge, toggle_edge(graph.edge_type(edge)))
        else:
            graph.set_type(vertex, VertexType.Z)


def _assert_reference_colors_restored(graph, reference_rg: ZXState) -> None:
    reference_index = _uid_index(reference_rg.graph)
    candidate_index = _uid_index(graph)
    for uid, reference_vertex in reference_index.items():
        candidate_vertex = candidate_index.get(uid)
        if candidate_vertex is None:
            raise ValueError(f"Missing pre-existing spider UID {uid}")
        if graph.type(candidate_vertex) != reference_rg.graph.type(reference_vertex):
            raise AssertionError(f"RG color changed for pre-existing spider UID {uid}")


def _choose_incident_qpu_owner(graph, vertex: int) -> int:
    owner_counts: dict[int, int] = {}
    for neighbor in graph.neighbors(vertex):
        if graph.type(neighbor) == VertexType.BOUNDARY:
            continue
        owner = _owner(graph, neighbor)
        owner_counts[owner] = owner_counts.get(owner, 0) + 1

    if not owner_counts:
        raise ValueError("Auxiliary spider has no owned internal neighbor")

    return min(owner_counts, key=lambda qpu: (-owner_counts[qpu], qpu))


def _assign_euler_metadata(graph, vertex: int, next_uid: int) -> int:
    owner = _choose_incident_qpu_owner(graph, vertex)
    graph.set_vdata(vertex, UID_KEY, next_uid)
    graph.set_vdata(vertex, QPU_KEY, owner)
    graph.set_vdata(vertex, ORIGIN_KEY, "lc_euler")
    return next_uid + 1


def _expand_residual_hadamard_edges_inplace(graph, next_uid: int) -> int:
    """Euler-expand LC-created X--H--X edges in place."""

    pending: list[tuple[int, int]] = []
    for edge in graph.edges():
        u, v = graph.edge_st(edge)
        if VertexType.BOUNDARY in (graph.type(u), graph.type(v)):
            continue
        if graph.edge_type(edge) == EdgeType.HADAMARD:
            pending.append(tuple(sorted((_uid(graph, u), _uid(graph, v)))))

    uid_index = _uid_index(graph)

    for uid_u, uid_v in sorted(set(pending)):
        u = uid_index.get(uid_u)
        v = uid_index.get(uid_v)
        if u is None or v is None or not graph.connected(u, v):
            continue

        edge = graph.edge(u, v)
        if graph.edge_type(edge) != EdgeType.HADAMARD:
            continue
        if graph.type(u) != VertexType.X or graph.type(v) != VertexType.X:
            raise ValueError(
                "Exact color restoration produced an unexpected internal H edge; "
                "expected only LC-created X--H--X edges"
            )

        before_vertices = set(graph.vertices())
        if not euler_expansion_rewrite.apply(graph, u, v):
            raise RuntimeError("PyZX failed to Euler-expand a residual H edge")

        new_vertices = set(graph.vertices()) - before_vertices
        if len(new_vertices) != 1:
            raise ValueError(
                "Expected exactly one Euler auxiliary for an X--H--X edge; "
                f"found {len(new_vertices)}"
            )
        new_vertex = next(iter(new_vertices))
        if graph.type(new_vertex) != VertexType.Z:
            raise ValueError("Euler expansion of X--H--X did not create a Z spider")

        next_uid = _assign_euler_metadata(
            graph,
            vertex=new_vertex,
            next_uid=next_uid,
        )
        uid_index[next_uid - 1] = new_vertex

    for edge in graph.edges():
        u, v = graph.edge_st(edge)
        if VertexType.BOUNDARY in (graph.type(u), graph.type(v)):
            continue
        if graph.edge_type(edge) != EdgeType.SIMPLE:
            raise ValueError("LC normalization left an internal H edge")
        if {graph.type(u), graph.type(v)} != {VertexType.X, VertexType.Z}:
            raise ValueError("LC normalization did not restore X--Z incidence")

    return next_uid


def graph_like_to_scoring_state(
    graph_like_state: ZXState,
    reference_rg: ZXState,
    center_uid: int,
    new_lc_uids: Sequence[int],
) -> ZXState:
    """Normalize one LC candidate back into hypergraph-scoring RG form."""

    graph = graph_like_state.graph
    _restore_reference_rg_colors(
        graph,
        reference_rg=reference_rg,
        center_uid=center_uid,
        new_lc_uids=new_lc_uids,
    )
    _assert_reference_colors_restored(graph, reference_rg)

    next_uid = _expand_residual_hadamard_edges_inplace(
        graph, graph_like_state.next_uid
    )
    next_uid = _extract_z_phases_inplace(graph, next_uid)

    state = ZXState(graph=graph, next_uid=next_uid)
    _validate_scoring_state(state)
    return state


# ============================================================
# Boundary-guided local complementation
# ============================================================


def lc_candidate_center_uids(
    graph_like_state: ZXState,
    boundary_x_uids: Sequence[int],
) -> tuple[int, ...]:
    """Neighbors of boundary X spiders, excluding neighbors also on boundary."""

    graph = graph_like_state.graph
    index = _uid_index(graph)
    boundary = set(boundary_x_uids)
    candidates: set[int] = set()

    for boundary_uid in boundary_x_uids:
        vertex = index.get(boundary_uid)
        if vertex is None:
            continue

        for neighbor in graph.neighbors(vertex):
            if graph.type(neighbor) == VertexType.BOUNDARY:
                continue

            neighbor_uid = _uid(graph, neighbor)
            if neighbor_uid in boundary:
                continue
            if is_lc_vertex(graph, neighbor):
                candidates.add(neighbor_uid)

    return tuple(sorted(candidates))


def _attach_lc_new_vertex_metadata(
    graph,
    new_vertices: Sequence[int],
    center_vertex: int,
    next_uid: int,
) -> tuple[int, tuple[int, ...]]:
    """Label the single local-Clifford gadget introduced by PyZX ``lc_cong``."""

    if len(new_vertices) != 1:
        raise ValueError(
            f"Expected lc_cong to introduce one gadget; found {len(new_vertices)}"
        )

    center_qpu = _owner(graph, center_vertex)
    vertex = new_vertices[0]
    graph.set_vdata(vertex, UID_KEY, next_uid)
    graph.set_vdata(vertex, QPU_KEY, center_qpu)
    graph.set_vdata(vertex, ORIGIN_KEY, "lc_gadget")
    return next_uid + 1, (next_uid,)


def optimize_with_local_complementation(
    initial_state: ZXState,
    safety_max_steps: int,
) -> tuple[ZXState, CutScore, tuple[LCRecord, ...]]:
    """Best-improvement LC search using only cut-hyperedge count.

    On every iteration:
      1. identify boundary X spiders from currently cut hyperedges;
      2. try LC at valid neighbors of those boundary vertices;
      3. normalize each candidate back into X/Z hypergraph form;
      4. accept the candidate with the smallest strict cut-count reduction.

    Induced local Clifford operations are recorded in ``LCRecord``. They are
    already represented by the equivalence-preserving PyZX rewrite and must not
    be appended again after circuit extraction.
    """

    current_state = initial_state
    current_score = score_hypergraph(build_hypergraph(current_state))
    history: list[LCRecord] = []

    for step in range(1, safety_max_steps + 1):
        if current_score.cost == 0:
            break

        reference_rg, graph_like_state = make_graph_like_for_lc(current_state)
        center_uids = lc_candidate_center_uids(
            graph_like_state,
            current_score.boundary_x_vertices,
        )
        if not center_uids:
            break

        best: tuple[int, int, ZXState, CutScore, LCRecord] | None = None

        for center_uid in center_uids:
            candidate_graph = graph_like_state.graph.copy()
            index = _uid_index(candidate_graph)
            center = index[center_uid]

            neighbor_uids = tuple(
                sorted(
                    _uid(candidate_graph, neighbor)
                    for neighbor in candidate_graph.neighbors(center)
                    if candidate_graph.type(neighbor) == VertexType.Z
                )
            )
            tracked_uids = {center_uid, *neighbor_uids}
            phases_before = _phase_by_uid(candidate_graph, tracked_uids, index=index)

            vertices_before_lc = set(candidate_graph.vertices())
            try:
                lc_cong(candidate_graph, center)
            except (ValueError, RuntimeError, AssertionError, KeyError):
                continue

            raw_new_vertices = tuple(
                sorted(set(candidate_graph.vertices()) - vertices_before_lc)
            )
            try:
                candidate_next_uid, new_uids = _attach_lc_new_vertex_metadata(
                    candidate_graph,
                    new_vertices=raw_new_vertices,
                    center_vertex=center,
                    next_uid=graph_like_state.next_uid,
                )
            except ValueError:
                continue

            index_after_lc = _uid_index(candidate_graph)
            phases_after = _phase_by_uid(
                candidate_graph, tracked_uids, index=index_after_lc
            )
            common_uids = phases_before.keys() & phases_after.keys()
            phase_changes = tuple(
                (uid, phases_before[uid], phases_after[uid])
                for uid in sorted(common_uids)
                if phases_before[uid] != phases_after[uid]
            )
            new_vertices = tuple(
                (
                    uid,
                    _owner(candidate_graph, index_after_lc[uid]),
                    str(candidate_graph.phase(index_after_lc[uid])),
                )
                for uid in new_uids
            )

            try:
                candidate_gh_state = ZXState(
                    graph=candidate_graph,
                    next_uid=candidate_next_uid,
                )
                candidate_state = graph_like_to_scoring_state(
                    candidate_gh_state,
                    reference_rg=reference_rg,
                    center_uid=center_uid,
                    new_lc_uids=new_uids,
                )
                candidate_score = score_hypergraph(build_hypergraph(candidate_state))
            except (ValueError, RuntimeError, AssertionError):
                continue

            # The requested figure of merit is the NUMBER of cut hyperedges.
            if candidate_score.cost >= current_score.cost:
                continue

            record = LCRecord(
                step=step,
                center_uid=center_uid,
                center_qpu=_owner(candidate_graph, center),
                neighbor_uids_before=neighbor_uids,
                cuts_before=current_score.cost,
                cuts_after=candidate_score.cost,
                cut_legs_before=current_score.cut_leg_count,
                cut_legs_after=candidate_score.cut_leg_count,
                theoretical_clifford_count=1 + len(neighbor_uids),
                center_clifford="exp(-i*pi*X/4) [up to convention/global phase]",
                neighbor_clifford="exp(+i*pi*Z/4) on each LC neighbor",
                phase_changes=phase_changes,
                new_vertices=new_vertices,
            )

            candidate = (
                candidate_score.cost,
                center_uid,
                candidate_state,
                candidate_score,
                record,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if best is None:
            break

        _, _, current_state, current_score, record = best
        if record.cuts_after >= record.cuts_before:
            raise AssertionError("Accepted LC did not strictly reduce cut count")
        history.append(record)

    else:
        raise RuntimeError(
            "LC search hit its safety iteration bound. Every accepted move "
            "strictly lowers an integer cut count, so inspect the rewrite logic "
            "before simply raising this bound."
        )

    return current_state, current_score, tuple(history)


# ============================================================
# Final extraction (only once) and equivalence checking
# ============================================================


def _load_pyzx_qasm2(program: str) -> QuantumCircuit:
    """Load OpenQASM 2 emitted by PyZX into Qiskit robustly.

    PyZX can emit operations such as ``swap`` after circuit extraction.
    Qiskit's strict OpenQASM-2 ``qelib1.inc`` follows the original spec and
    does not define all gates historically treated as built-ins by Qiskit.
    ``LEGACY_CUSTOM_INSTRUCTIONS`` supplies those compatibility definitions,
    including ``swap`` -> ``SwapGate``.

    We prefer the full compatibility table because future extraction results
    can contain other legacy-standard operations as well.
    """

    legacy_instructions = getattr(qasm2, "LEGACY_CUSTOM_INSTRUCTIONS", ())

    if legacy_instructions:
        return qasm2.loads(
            program,
            custom_instructions=legacy_instructions,
        )

    # Defensive fallback for Qiskit versions that expose CustomInstruction
    # but not LEGACY_CUSTOM_INSTRUCTIONS.
    from qiskit.circuit.library import SwapGate

    swap_instruction = qasm2.CustomInstruction(
        "swap",
        0,
        2,
        SwapGate,
        builtin=True,
    )
    return qasm2.loads(
        program,
        custom_instructions=(swap_instruction,),
    )


def extract_final_circuit(state: ZXState) -> QuantumCircuit:
    """Extract one ordinary circuit after LC optimization has terminated."""

    fused = _fuse_phase_carriers(state)
    graph = fused.graph.copy()
    full_reduce(graph)

    pyzx_circuit = extract_circuit(
        graph,
        optimize_czs=True,
        optimize_cnots=2,
        up_to_perm=False,
        quiet=True,
    )

    # Do not parse PyZX's QASM with Qiskit's strict defaults.  In particular,
    # extracted circuits can contain ``swap``, which is not part of the strict
    # OpenQASM-2 qelib1.inc understood by Qiskit's new parser.
    return _load_pyzx_qasm2(
        pyzx_circuit.to_qasm(version=2)
    )


def verify_equivalence(original: QuantumCircuit, optimized: QuantumCircuit) -> bool | None:
    original_pyzx = zx.Circuit.from_qasm(qasm2.dumps(original))
    optimized_pyzx = zx.Circuit.from_qasm(qasm2.dumps(optimized))
    return original_pyzx.verify_equality(
        optimized_pyzx,
        up_to_global_phase=True,
    )


# ============================================================
# One trial
# ============================================================


def run_trial(
    source_kind: str,
    requested_num_qubits: int,
    trial: int,
    config: ExperimentConfig,
    output_dir: Path,
    benchmark: str | None = None,
) -> TrialResult:
    """Run one source circuit through warm-start partitioning and LC search."""

    source_stream = 10 if source_kind == "random" else 20
    benchmark_stream = 0 if benchmark is None else sum(ord(c) for c in benchmark) % 997

    circuit_seed = derived_seed(
        config.master_seed,
        requested_num_qubits,
        trial,
        stream=source_stream + benchmark_stream,
    )
    transpiler_seed = derived_seed(
        config.master_seed,
        requested_num_qubits,
        trial,
        stream=1 + benchmark_stream,
    )
    partition_seed = derived_seed(
        config.master_seed,
        requested_num_qubits,
        trial,
        stream=2 + benchmark_stream,
    )

    source = make_source_circuit(
        source_kind=source_kind,
        requested_num_qubits=requested_num_qubits,
        trial=trial,
        config=config,
        circuit_seed=circuit_seed,
        benchmark=benchmark,
    )

    # MQT documents circuit_size as the qubit count in most cases, but use the
    # actual generated width from here onward.
    num_qubits = source.num_qubits
    compiled = compile_to_rx_rz_cx(source, transpiler_seed)

    prepared_state = prepare_initial_zx(circuit_to_zx(compiled), num_qubits)
    qpus = qpus_for_size(num_qubits, config)

    initial_state, warm_start = choose_warm_start(
        prepared_state=prepared_state,
        num_qubits=num_qubits,
        qpus=qpus,
        num_samples=config.random_partitions,
        seed=partition_seed,
    )

    optimized_state, optimized_score, lc_history = optimize_with_local_complementation(
        initial_state,
        safety_max_steps=config.safety_max_lc_steps,
    )

    if optimized_score.cost > warm_start.score.cost:
        raise AssertionError("LC optimization increased the cut-hyperedge objective")

    # Per the requested workflow, this is the only ZX -> circuit extraction.
    optimized_circuit = extract_final_circuit(optimized_state)
    if optimized_circuit.num_qubits != compiled.num_qubits:
        raise RuntimeError(
            f"Circuit extraction changed qubit count: {compiled.num_qubits} -> "
            f"{optimized_circuit.num_qubits}"
        )

    equivalence: bool | None = None
    if config.verify_equivalence:
        equivalence = verify_equivalence(compiled, optimized_circuit)
        if equivalence is not True:
            raise RuntimeError(
                f"Equivalence verification failed/inconclusive for "
                f"{source_kind}:{benchmark or 'random'}, "
                f"{num_qubits}q trial {trial}: {equivalence}"
            )

    source_label = "random" if source_kind == "random" else f"mqt_{benchmark}"
    trial_dir = (
        output_dir
        / "circuits"
        / source_label
        / f"{requested_num_qubits}q"
        / f"trial_{trial:02d}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    (trial_dir / "source.qasm").write_text(qasm2.dumps(source), encoding="utf-8")
    (trial_dir / "baseline_rx_rz_cx.qasm").write_text(
        qasm2.dumps(compiled), encoding="utf-8"
    )
    (trial_dir / "optimized_zx_lc.qasm").write_text(
        qasm2.dumps(optimized_circuit), encoding="utf-8"
    )
    (trial_dir / "logical_partition.json").write_text(
        json.dumps(
            {
                "source_kind": source_kind,
                "benchmark": benchmark,
                "requested_num_qubits": requested_num_qubits,
                "actual_num_qubits": num_qubits,
                "logical_to_qpu": warm_start.logical_to_qpu,
                "sampled_cut_counts": warm_start.sampled_objectives,
                "chosen_cut_hyperedges": warm_start.score.cost,
                "chosen_cut_z_legs": warm_start.score.cut_leg_count,
                "qpus": [asdict(qpu) for qpu in qpus],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (trial_dir / "lc_history.json").write_text(
        json.dumps([asdict(record) for record in lc_history], indent=2),
        encoding="utf-8",
    )

    induced_cliffords = sum(
        record.theoretical_clifford_count for record in lc_history
    )

    return TrialResult(
        source_kind=source_kind,
        benchmark=benchmark or "random",
        requested_num_qubits=requested_num_qubits,
        num_qubits=num_qubits,
        trial=trial,
        circuit_seed=circuit_seed,
        transpiler_seed=transpiler_seed,
        partition_seed=partition_seed,
        baseline_cut_hyperedges=warm_start.score.cost,
        optimized_cut_hyperedges=optimized_score.cost,
        baseline_cut_z_legs=warm_start.score.cut_leg_count,
        optimized_cut_z_legs=optimized_score.cut_leg_count,
        accepted_lcs=len(lc_history),
        induced_cliffords=induced_cliffords,
        equivalence_verified=equivalence,
    )


# ============================================================
# Aggregation and plotting
# ============================================================


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def summarize_results(
    results: Sequence[TrialResult],
) -> list[dict[str, float | int | str]]:
    """Summarize each source/benchmark family separately by actual width."""

    summary: list[dict[str, float | int | str]] = []

    keys = sorted(
        {(result.source_label, result.num_qubits) for result in results}
    )

    for source_label, num_qubits in keys:
        group = [
            result
            for result in results
            if result.source_label == source_label and result.num_qubits == num_qubits
        ]

        baseline_cut_mean, baseline_cut_std = mean_std(
            [result.baseline_cut_hyperedges for result in group]
        )
        optimized_cut_mean, optimized_cut_std = mean_std(
            [result.optimized_cut_hyperedges for result in group]
        )
        baseline_leg_mean, baseline_leg_std = mean_std(
            [result.baseline_cut_z_legs for result in group]
        )
        optimized_leg_mean, optimized_leg_std = mean_std(
            [result.optimized_cut_z_legs for result in group]
        )
        accepted_lc_mean, accepted_lc_std = mean_std(
            [result.accepted_lcs for result in group]
        )
        reduction_mean, reduction_std = mean_std(
            [result.cut_reduction_fraction for result in group]
        )

        summary.append(
            {
                "source": source_label,
                "num_qubits": num_qubits,
                "num_circuits": len(group),
                "baseline_cut_mean": baseline_cut_mean,
                "baseline_cut_std": baseline_cut_std,
                "optimized_cut_mean": optimized_cut_mean,
                "optimized_cut_std": optimized_cut_std,
                "baseline_cut_z_legs_mean": baseline_leg_mean,
                "baseline_cut_z_legs_std": baseline_leg_std,
                "optimized_cut_z_legs_mean": optimized_leg_mean,
                "optimized_cut_z_legs_std": optimized_leg_std,
                "accepted_lcs_mean": accepted_lc_mean,
                "accepted_lcs_std": accepted_lc_std,
                "cut_reduction_fraction_mean": reduction_mean,
                "cut_reduction_fraction_std": reduction_std,
            }
        )

    return summary


def save_trial_csv(results: Sequence[TrialResult], path: Path) -> None:
    if not results:
        return

    rows = []
    for result in results:
        row = asdict(result)
        row["source_label"] = result.source_label
        row["cut_reduction_fraction"] = result.cut_reduction_fraction
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(
    summary: Sequence[dict[str, float | int | str]],
    path: Path,
) -> None:
    if not summary:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def plot_cut_hyperedges(
    summary: Sequence[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    """Create one warm-start-vs-LC cut plot per circuit family."""

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted({str(row["source"]) for row in summary})

    for source in sources:
        rows = sorted(
            (row for row in summary if str(row["source"]) == source),
            key=lambda row: int(row["num_qubits"]),
        )
        if not rows:
            continue

        x = np.arange(len(rows), dtype=float)
        width = 0.36

        baseline = [float(row["baseline_cut_mean"]) for row in rows]
        optimized = [float(row["optimized_cut_mean"]) for row in rows]
        baseline_std = [float(row["baseline_cut_std"]) for row in rows]
        optimized_std = [float(row["optimized_cut_std"]) for row in rows]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(
            x - width / 2,
            baseline,
            width,
            yerr=baseline_std,
            capsize=4,
            label="Warm start",
        )
        ax.bar(
            x + width / 2,
            optimized,
            width,
            yerr=optimized_std,
            capsize=4,
            label="After LC",
        )
        ax.set_xticks(x, [str(row["num_qubits"]) for row in rows])
        ax.set_xlabel("Logical qubits")
        ax.set_ylabel("Average cut hyperedges")
        ax.set_title(f"Distributed hypergraph cuts: {source}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"cut_hyperedges_{_safe_filename(source)}.png",
            dpi=180,
        )
        plt.close(fig)


# ============================================================
# Complete benchmark
# ============================================================


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("pyzx", "qiskit", "mqt-bench", "numpy", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def validate_config(config: ExperimentConfig) -> None:
    if config.num_qpus < 1:
        raise ValueError("num_qpus must be positive")
    if not config.run_random_circuits and not config.run_mqt_benchmarks:
        raise ValueError("At least one circuit source must be enabled")
    if config.run_random_circuits and config.circuits_per_size < 1:
        raise ValueError("circuits_per_size must be positive")
    if config.run_mqt_benchmarks and config.mqt_repetitions < 1:
        raise ValueError("mqt_repetitions must be positive")
    if config.run_mqt_benchmarks and not config.mqt_benchmarks:
        raise ValueError("mqt_benchmarks cannot be empty when MQT mode is enabled")
    if config.random_partitions < 1:
        raise ValueError("random_partitions must be positive")
    if config.safety_max_lc_steps < 1:
        raise ValueError("safety_max_lc_steps must be positive")
    if not 0.0 <= config.single_qubit_gate_probability <= 1.0:
        raise ValueError("single_qubit_gate_probability must be in [0,1]")
    if not 0.0 <= config.cx_probability <= 1.0:
        raise ValueError("cx_probability must be in [0,1]")

    for num_qubits in config.qubit_sizes:
        qpus_for_size(num_qubits, config)

    if config.run_mqt_benchmarks and (get_benchmark is None or BenchmarkLevel is None):
        raise ImportError(
            "MQT mode is enabled but mqt-bench is not installed. Install with:\n"
            "    pip install mqt-bench"
        )


def run_benchmark(config: ExperimentConfig = ExperimentConfig()) -> list[TrialResult]:
    """Run random circuits and MQT Bench through one identical ZX-LC pipeline."""

    validate_config(config)
    seed_process_globals(config.master_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrialResult] = []

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": asdict(config),
                "package_versions": package_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for requested_num_qubits in config.qubit_sizes:
        # ----------------------------------------------------
        # Reproducible synthetic random circuits
        # ----------------------------------------------------
        if config.run_random_circuits:
            for trial in range(config.circuits_per_size):
                print(
                    f"[random] {requested_num_qubits}q "
                    f"trial {trial + 1}/{config.circuits_per_size}"
                )
                result = run_trial(
                    source_kind="random",
                    requested_num_qubits=requested_num_qubits,
                    trial=trial,
                    config=config,
                    output_dir=output_dir,
                )
                results.append(result)
                save_trial_csv(results, output_dir / "trial_results.csv")
                print(
                    f"  cuts {result.baseline_cut_hyperedges} -> "
                    f"{result.optimized_cut_hyperedges}, "
                    f"LCs={result.accepted_lcs}, "
                    f"Cliffords={result.induced_cliffords}"
                )

        # ----------------------------------------------------
        # MQT Bench algorithm-level circuits
        # ----------------------------------------------------
        if config.run_mqt_benchmarks:
            for benchmark in config.mqt_benchmarks:
                for trial in range(config.mqt_repetitions):
                    print(
                        f"[MQT:{benchmark}] {requested_num_qubits}q "
                        f"instance {trial + 1}/{config.mqt_repetitions}"
                    )
                    try:
                        result = run_trial(
                            source_kind="mqt",
                            benchmark=benchmark,
                            requested_num_qubits=requested_num_qubits,
                            trial=trial,
                            config=config,
                            output_dir=output_dir,
                        )
                    except Exception as exc:
                        if not config.skip_failed_mqt:
                            raise
                        print(f"  SKIPPED: {type(exc).__name__}: {exc}")
                        continue

                    results.append(result)
                    save_trial_csv(results, output_dir / "trial_results.csv")
                    print(
                        f"  cuts {result.baseline_cut_hyperedges} -> "
                        f"{result.optimized_cut_hyperedges}, "
                        f"LCs={result.accepted_lcs}, "
                        f"Cliffords={result.induced_cliffords}"
                    )

    summary = summarize_results(results)
    save_trial_csv(results, output_dir / "trial_results.csv")
    save_summary_csv(summary, output_dir / "summary.csv")
    plot_cut_hyperedges(summary, output_dir)

    print(f"\nResults written to: {output_dir.resolve()}")
    return results


if __name__ == "__main__":
    CONFIG = ExperimentConfig(
        qubit_sizes=(4, 8),
        run_random_circuits=True,
        circuits_per_size=10,
        circuit_layers=10,
        run_mqt_benchmarks=True,
        mqt_benchmarks=("qft", "qftentangled", "graphstate", "qaoa", "grover"),
        mqt_repetitions=1,
        num_qpus=2,
        random_partitions=10,
        safety_max_lc_steps=1000,
        master_seed=20260901,
        verify_equivalence=False,
        skip_failed_mqt=True,
        output_dir="zx_lc_random_mqt_results",
    )

    run_benchmark(CONFIG)
