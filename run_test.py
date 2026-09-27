import matplotlib.pyplot as plt
import environment


def run_simulation(handover_policy="load_aware", total_timesteps=60):
    """
    Runs a multi-RAT mobility scenario.
    handover_policy: 'rsrp' (baseline) or 'load_aware' (proposed QoS-aware adaptation)
    """
    if handover_policy not in {"rsrp", "load_aware"}:
        raise ValueError("handover_policy must be 'rsrp' or 'load_aware'")

    # util.py resolves stations through these class-level lists.
    environment.wireless_environment.bs_list = []
    environment.wireless_environment.ue_list = []

    # Initialize environment (width=1000m, height=1000m, sampling_time=1s).
    env = environment.wireless_environment(1000, 1000, 1)

    # Frequencies are MHz and bitrates are Mbps in this repository.
    env.place_SAT_base_station(total_bitrate=50, position=(500, 500))

    env.place_LTE_base_station(
        position=(300, 500, 40),
        carrier_frequency=1800,
        antenna_power=40,
        antenna_gain=15,
        feeder_loss=2,
        available_bandwidth=20,
        total_bitrate=100
    )

    env.place_NR_base_station(
        position=(700, 500, 40),
        carrier_frequency=1900,
        numerology=1,
        antenna_power=30,
        antenna_gain=12,
        feeder_loss=1,
        available_bandwidth=50,
        total_bitrate=300
    )

    # Insert a moving UE traveling horizontally across the cells.
    ue_id = env.insert_ue(
        ue_class=0,
        starting_position=(100, 500, 1),
        speed=15,
        direction=0
    )

    requested_bitrate = 15
    active_bs = None
    
    # Tracking metrics
    history = {
        "timestep": [],
        "active_bs": [],
        "delivered_bitrate": [],
        "handover_events": 0
    }

    env.initial_timestep()

    # 4. Step-by-step simulation loop
    for t in range(total_timesteps):
        # Discover visible base stations and their RSRP
        rsrp_dict = env.discover_bs(ue_id)
        if not rsrp_dict:
            env.next_timestep()
            continue

        selected_bs = None

        if handover_policy == "rsrp":
            # Baseline: Connect purely to highest raw receiving power
            selected_bs = max(rsrp_dict, key=rsrp_dict.get)

        elif handover_policy == "load_aware":
            # Proposed: Multi-factor score considering RSRP and Base Station congestion
            best_score = -float("inf")
            for bs_candidate, rsrp_val in rsrp_dict.items():
                ap = env.bs_list[bs_candidate]
                rbur = ap.compute_rbur()  # Resource Block Utilization Ratio (0.0 to 1.0)
                
                # Penalize congested base stations and add hysteresis to avoid ping-ponging
                score = rsrp_val - (rbur * 20.0)
                if bs_candidate == active_bs:
                    score += 3.0  # Hysteresis bonus to maintain connection stability

                if score > best_score:
                    best_score = score
                    selected_bs = bs_candidate

        # Execute connection, handover, or link update
        allocated_rate = 0
        if active_bs is None:
            # Initial connection
            allocated_rate = env.bs_list[selected_bs].request_connection(ue_id, requested_bitrate, rsrp_dict)
            active_bs = selected_bs
        elif selected_bs != active_bs:
            # Seamless Handover execution: detach from old, attach to new
            env.bs_list[active_bs].request_disconnection(ue_id)
            allocated_rate = env.bs_list[selected_bs].request_connection(ue_id, requested_bitrate, rsrp_dict)
            active_bs = selected_bs
            history["handover_events"] += 1
        else:
            # Keep link alive and recalculate resources given movement/fading
            allocated_rate = env.bs_list[active_bs].update_connection(ue_id, requested_bitrate, rsrp_dict)

        # AP request/update methods return Mbps already.
        history["timestep"].append(t)
        history["active_bs"].append(active_bs)
        history["delivered_bitrate"].append(allocated_rate)

        env.next_timestep()

    return history


def plot_comparison(baseline, improved):
    """Plot throughput for the baseline and load-aware policies."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        baseline["timestep"],
        baseline["delivered_bitrate"],
        label=f"Baseline RSRP (Handovers: {baseline['handover_events']})",
        linestyle="--",
        color="crimson",
    )
    ax.plot(
        improved["timestep"],
        improved["delivered_bitrate"],
        label=f"Load-Aware Policy (Handovers: {improved['handover_events']})",
        linewidth=2,
        color="teal",
    )
    ax.axhline(y=15, color="gray", linestyle=":", label="Requested Target (15 Mbps)")
    ax.set_title("QoS Stability Across Heterogeneous Handovers (Satellite / LTE / NR)")
    ax.set_xlabel("Simulation Timestep (seconds)")
    ax.set_ylabel("Delivered Throughput (Mbps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    baseline = run_simulation(handover_policy="rsrp")
    improved = run_simulation(handover_policy="load_aware")
    plot_comparison(baseline, improved)
    plt.show()