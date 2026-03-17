Tips
====

Determining the rendezvous IP address and port
----------------------------------------------

When launching a simulation across multiple ranks, it is often convenient to
have rank 0 choose the Gloo rendezvous address and port, then share them with
the other ranks through a small JSON file.

In the example below, rank 0 extracts the IPv4 address of the high-speed
network interface named ``hsn0``, binds a socket to port ``0`` to let the OS
pick a free port, and writes both values to ``gloo_rdzv.json``. All other
ranks wait until that file exists, then read it and reuse the same rendezvous
endpoint.

.. code-block:: python

    rdzv = "./gloo_rdzv.json"
    if rank == 0:
        ip = subprocess.check_output(
            "ip -o -4 addr show hsn0 | awk '{print $4}' | cut -d/ -f1",
            shell=True, text=True
        ).strip()
        s = socket.socket(); s.bind((ip, 0))
        port = s.getsockname()[1]; s.close()
        json.dump({"ip": ip, "port": port}, open(rdzv, "w"))
    else:
        while not os.path.exists(rdzv): time.sleep(0.1)
    data = json.load(open(rdzv))
    master_addr, master_port = data["ip"], data["port"]
