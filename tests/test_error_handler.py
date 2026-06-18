import pytest
import ray
import numpy as np
from deisa.ray.errors import ContractError
from deisa.ray.bridge import Bridge
from tests.comm_utils import NoOpComm
from tests.utils import ray_multinode_cluster, wait_for_head_node  # noqa: F401
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(max_retries=0)
def head_script() -> None:
    """The head node checks that the values are correct"""
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    # TODO : modify assert to test the actual error handler
    @d.register("array")
    def simulation_callback(array):
        assert False

    d.execute_callbacks()


@ray.remote(max_retries=0)
def bridge_script(rank: int) -> str:
    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(rank, 2),
    )  # type:ignore

    bridge.send(
        array_name="array",
        chunk=np.array([[rank + 1]], dtype=np.int64),
        timestep=0,
    )

    return bridge.node_id


@ray.remote(max_retries=0)
def contract_head_script() -> None:
    from deisa.ray.window_handler import Deisa

    d = Deisa()

    @d.register("array")
    def simulation_callback(array):
        pass

    d.execute_callbacks()


@ray.remote(max_retries=0)
def contract_error_bridge_script(rank: int) -> str:
    arrays_md = {
        "array": {
            "global_shape": (1, 2),
            "chunk_shape": (1, 1),
            "chunk_position": (0, rank),
        }
    }

    bridge = Bridge(
        arrays_metadata=arrays_md,
        comm=NoOpComm(rank, 2),
    )  # type:ignore

    bridge.send(
        array_name="not_described",
        chunk=np.array([[rank + 1]], dtype=np.int64),
        timestep=0,
    )

    return bridge.node_id


# CRITICAL WARNING : This test checks that an assertion error in the callback is detected. If this test fails,
# it means that all callbacks could secretely fail and the test harness is not detecting it.
def test_exception_handler_not_bypass_computation(ray_multinode_cluster) -> None:  # noqa: F811
    with pytest.raises(AssertionError):
        cluster = ray_multinode_cluster["cluster"]
        head_node_id = None
        worker_node_ids = []
        for node in cluster.list_all_nodes():
            if node.is_head():
                head_node_id = node.node_id
            else:
                worker_node_ids.append(node.node_id)

        head_ref = head_script.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head_node_id,
                soft=False,
            ),
        ).remote()
        wait_for_head_node()

        ray.get(
            bridge_script.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[1],
                    soft=False,
                ),
            ).remote(1)
        )
        ray.get(
            bridge_script.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[0],
                    soft=False,
                ),
            ).remote(0)
        )
        ray.get(head_ref)


def test_contract_error(ray_multinode_cluster) -> None:  # noqa: F811
    cluster = ray_multinode_cluster["cluster"]
    head_node_id = None
    worker_node_ids = []
    for node in cluster.list_all_nodes():
        if node.is_head():
            head_node_id = node.node_id
        else:
            worker_node_ids.append(node.node_id)

    head_ref = contract_head_script.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote()
    wait_for_head_node()

    with pytest.raises(ContractError):
        ray.get(
            contract_error_bridge_script.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[0],
                    soft=False,
                ),
            ).remote(0)
        )

    ray.cancel(head_ref, force=True)
