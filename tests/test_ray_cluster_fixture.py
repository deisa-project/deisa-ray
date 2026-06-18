from tests.utils import (
    pick_free_port,
    start_ray_multinode_cluster,
)
def test_ray_multinode_clusters_can_start_in_parallel():
    clusters = [
        start_ray_multinode_cluster(head_node_gcs_server_port=pick_free_port()),
        start_ray_multinode_cluster(head_node_gcs_server_port=pick_free_port()),
    ]

    try:
        addresses = [cluster.address for cluster in clusters]
        assert len(set(addresses)) == len(addresses)
    finally:
        for cluster in clusters:
            cluster.shutdown()