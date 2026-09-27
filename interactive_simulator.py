"""Interactive 2D visualizer for the wireless-network-simulator project.

Run from this directory with:
    python interactive_simulator.py

Click the map to place UEs. Use "Add base station" to create a station with
random type and parameters, then click Start to run the connection simulation.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button

import environment


class InteractiveSimulator:
    """Small interactive front end around the repository's simulation model."""

    def __init__(self, width: int = 4000, height: int = 4000) -> None:
        # bs_list and ue_list are class-level in the original project.
        environment.wireless_environment.bs_list = []
        environment.wireless_environment.ue_list = []

        self.env = environment.wireless_environment(width, height, sampling_time=1)
        self.requested_bitrate = 15
        self.running = False
        self.initialized = False
        self.active_bs: Dict[int, Optional[int]] = {}
        self.visible_bs: Dict[int, Dict[int, float]] = {}
        self.selected_ue: Optional[int] = None
        self.selected_bs: Optional[int] = None
        self.connection_rates: Dict[int, float] = {}
        self.station_scores: Dict[int, Dict[int, float]] = {}
        self.handover_reason: Dict[int, str] = {}
        self.step_number = 0

        self.figure, self.ax = plt.subplots(figsize=(10, 7))
        self.figure.subplots_adjust(bottom=0.16, right=0.78)
        self.status_ax = self.figure.add_axes((0.80, 0.16, 0.18, 0.70))
        self.status_ax.axis("off")

        self.timer = self.figure.canvas.new_timer(interval=500)
        self.timer.add_callback(self.step)
        self.figure.canvas.mpl_connect("button_press_event", self.on_map_click)

        self._create_controls()
        self._add_default_stations()
        self._draw()

    def _create_controls(self) -> None:
        button_specs = [
            ("Add base station", 0.03, self._add_random_station),
            ("Start", 0.27, self.start),
            ("Stop", 0.40, self.stop),
            ("Reset", 0.53, self.reset),
        ]
        self.buttons = []
        for label, x, callback in button_specs:
            button_ax = self.figure.add_axes((x, 0.04, 0.20 if x == 0.03 else 0.10, 0.06))
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def _add_default_stations(self) -> None:
        self.env.place_LTE_base_station(
            position=(800, 2000, 40),
            carrier_frequency=1800,
            antenna_power=40,
            antenna_gain=15,
            feeder_loss=2,
            available_bandwidth=20,
            total_bitrate=1000,
        )

        self.env.place_NR_base_station(
            position=(2000, 2000, 40),
            carrier_frequency=1900,
            numerology=1,
            antenna_power=30,
            antenna_gain=12,
            feeder_loss=1,
            available_bandwidth=50,
            total_bitrate=3000,
        )

        self.env.place_SAT_base_station(
            total_bitrate=5000,
            position=(3200, 2000),
        )

    def _add_random_station(self, _event=None) -> None:
        """Create an LTE, NR, or satellite station with valid random values."""
        if self.running:
            return

        station_type = random.choice(("LTE", "NR", "SAT"))
        x = random.uniform(150, self.env.x_limit - 150)
        y = random.uniform(150, self.env.y_limit - 150)

        if station_type == "SAT":
            station_id = self.env.place_SAT_base_station(
                total_bitrate=random.choice((1000, 5000, 10000)),
                position=(x, y),
            )
        elif station_type == "LTE":
            station_id = self.env.place_LTE_base_station(
                position=(x, y, random.uniform(30, 150)),
                carrier_frequency=random.choice((800, 1800, 1900)),
                antenna_power=random.choice((10, 20, 40)),
                antenna_gain=random.choice((5, 10, 16)),
                feeder_loss=random.choice((1, 2, 3)),
                available_bandwidth=random.choice((5, 10, 20)),
                total_bitrate=random.choice((500, 1000, 2000)),
            )
        else:
            station_id = self.env.place_NR_base_station(
                position=(x, y, random.uniform(30, 150)),
                # NRBaseStation also applies the COST-HATA 150-2000 MHz
                # constraint, even though environment.py permits up to 6 GHz.
                carrier_frequency=random.choice((700, 1800, 1900)),
                numerology=1,
                antenna_power=random.choice((10, 20, 30)),
                antenna_gain=random.choice((5, 10, 12)),
                feeder_loss=random.choice((1, 2)),
                available_bandwidth=random.choice((20, 40, 50)),
                total_bitrate=random.choice((500, 1000, 3000)),
            )

        self._draw(message=f"Added {station_type} base station {station_id}")

    def on_map_click(self, event) -> None:
        """Select a nearby UE or place a new one on the map."""
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        for station in self.env.bs_list:
            distance = math.hypot(
                station.position[0] - event.xdata,
                station.position[1] - event.ydata,
            )
            if distance <= 140:
                self.selected_bs = station.bs_id
                self.selected_ue = None
                self._draw(message=f"Selected BS {station.bs_id}")
                return

        for ue in self.env.ue_list:
            distance = (
                (ue.current_position[0] - event.xdata) ** 2
                + (ue.current_position[1] - event.ydata) ** 2
            ) ** 0.5
            if distance <= 120:
                self.selected_ue = ue.ue_id
                self.selected_bs = None
                self._draw(message=f"Selected UE {ue.ue_id}")
                return

        if self.running:
            return

        ue_id = self.env.insert_ue(
            ue_class=0,
            starting_position=(event.xdata, event.ydata, 1),
            speed=50,
            direction=0,
        )
        self.active_bs[ue_id] = None
        self.selected_ue = ue_id
        self.selected_bs = None
        self._draw(message=f"Placed UE {ue_id}; click Start to connect it")

    def start(self, _event=None) -> None:
        if self.running:
            return
        if not self.env.bs_list:
            self._draw(message="Add at least one base station first")
            return
        if not self.env.ue_list:
            self._draw(message="Click the map to place a UE first")
            return

        try:
            if not self.initialized:
                self.env.initial_timestep()
                self.initialized = True
            self.running = True
            self.timer.start()
            self._draw(message="Simulation running")
        except Exception as exc:
            self._draw(message=f"Could not start: {exc}")

    def stop(self, _event=None) -> None:
        self.running = False
        self.timer.stop()
        self._draw(message="Simulation stopped")

    def reset(self, _event=None):
        self.stop()
        width, height = self.env.x_limit, self.env.y_limit
        plt.close(self.figure)
        self.__init__(width, height)
    
    def _select_station(self, ue_id: int, rsrp: Dict[int, float]) -> int:
        """Select a station using signal, distance, load, and UE performance."""
        ue = next(ue for ue in self.env.ue_list if ue.ue_id == ue_id)
        current = self.active_bs.get(ue_id)
        map_diagonal = math.hypot(self.env.x_limit, self.env.y_limit)
        scores = {}

        for station_id, signal in rsrp.items():
            station = self.env.bs_list[station_id]

            distance = math.hypot(
                ue.current_position[0] - station.position[0],
                ue.current_position[1] - station.position[1],
            )
            distance_penalty = 8 * distance / map_diagonal

            load = station.allocated_bitrate / station.total_bitrate
            load = min(max(load, 0.0), 1.0)
            utilization = min(max(station.compute_rbur(), 0.0), 1.0)
            available_capacity = 1.0 - load

            score = (
                signal
                - distance_penalty
                - 12 * load
                - 8 * utilization
                + 8 * available_capacity
            )

            if station_id == current:
                current_rate = self.connection_rates.get(ue_id, 0.0)
                performance = current_rate / self.requested_bitrate
                performance = min(max(performance, 0.0), 1.0)
                score += 3 * performance
                score -= 12 * (1 - performance)

            scores[station_id] = score

        self.station_scores[ue_id] = scores
        return max(scores, key=scores.get)

    def _update_ue(self, ue_id: int) -> None:
        rsrp = self.env.discover_bs(ue_id)
        self.visible_bs[ue_id] = rsrp
        if not rsrp:
            return

        selected = self._select_station(ue_id, rsrp)
        current = self.active_bs.get(ue_id)
        if current is None:
            allocated_rate = self.env.bs_list[selected].request_connection(
                ue_id, self.requested_bitrate, rsrp
            )
        elif selected != current:
            self.env.bs_list[current].request_disconnection(ue_id)
            allocated_rate = self.env.bs_list[selected].request_connection(
                ue_id, self.requested_bitrate, rsrp
            )
            self.handover_reason[ue_id] = f"Switched BS {current} -> BS {selected}"
        else:
            allocated_rate = self.env.bs_list[current].update_connection(
                ue_id, self.requested_bitrate, rsrp
            )

        self.connection_rates[ue_id] = allocated_rate
        self.active_bs[ue_id] = selected

    def step(self, *_args) -> None:
        if not self.running:
            return
        try:
            for ue in self.env.ue_list:
                self._update_ue(ue.ue_id)
            self.env.next_timestep()
            self.step_number += 1
            self._draw()
        except Exception as exc:
            self.stop()
            self._draw(message=f"Simulation error: {exc}")

    def _draw(self, message: str = "") -> None:
        self.ax.clear()
        self.ax.set_xlim(0, self.env.x_limit)
        self.ax.set_ylim(0, self.env.y_limit)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x position (m)")
        self.ax.set_ylabel("y position (m)")
        self.ax.set_title("Interactive Wireless Network Simulator")

        colors = {"sat": "darkorange", "lte": "royalblue", "nr": "seagreen"}
        for station in self.env.bs_list:
            x, y = station.position
            color = colors.get(station.bs_type, "purple")
            selected = station.bs_id == self.selected_bs
            self.ax.scatter(
                x,
                y,
                s=280 if selected else 180,
                marker="^",
                color="gold" if selected else color,
                edgecolor="red" if selected else "black",
                linewidth=2 if selected else 1,
                zorder=3,
            )
            self.ax.text(x + 35, y + 35, f"BS {station.bs_id}\n{station.bs_type.upper()}", fontsize=8)

        for ue in self.env.ue_list:
            x, y = ue.current_position
            selected = ue.ue_id == self.selected_ue
            self.ax.scatter(
                x,
                y,
                s=180 if selected else 100,
                marker="o",
                color="gold" if selected else "black",
                edgecolor="red" if selected else "black",
                zorder=5,
            )
            self.ax.text(x + 35, y - 45, f"UE {ue.ue_id}", fontsize=8)
            for station_id in self.visible_bs.get(ue.ue_id, {}):
                station = self.env.bs_list[station_id]
                is_active = station_id == self.active_bs.get(ue.ue_id)
                self.ax.plot(
                    [x, station.position[0]], [y, station.position[1]],
                    color="crimson" if is_active else "gray",
                    linewidth=2 if is_active else 0.8,
                    linestyle="-" if is_active else ":",
                    alpha=0.9 if is_active else 0.35,
                    zorder=1,
                )

        self.ax.legend(handles=[
            Line2D([], [], marker="^", color="w", markerfacecolor="royalblue", label="LTE", markersize=10),
            Line2D([], [], marker="^", color="w", markerfacecolor="seagreen", label="NR", markersize=10),
            Line2D([], [], marker="^", color="w", markerfacecolor="darkorange", label="Satellite", markersize=10),
            Line2D([], [], marker="o", color="black", label="UE", markersize=7),
            Line2D([], [], color="crimson", label="Active connection"),
        ], loc="upper right", fontsize=8)

        lines = [message or ("Running" if self.running else "Stopped"), f"Step: {self.step_number}", ""]
        for ue in self.env.ue_list:
            station_id = self.active_bs.get(ue.ue_id)
            rate = self.connection_rates.get(ue.ue_id, 0.0)
            lines.append(f"UE {ue.ue_id}: BS {station_id}, {rate:.1f} Mbps")

        if self.selected_ue is not None:
            selected_ue = next(
                (ue for ue in self.env.ue_list if ue.ue_id == self.selected_ue),
                None,
            )
            if selected_ue is not None:
                station_id = self.active_bs.get(selected_ue.ue_id)
                rsrp = self.visible_bs.get(selected_ue.ue_id, {})
                lines.extend([
                    "",
                    f"Selected UE: {selected_ue.ue_id}",
                    f"Position: ({selected_ue.current_position[0]:.1f}, "
                    f"{selected_ue.current_position[1]:.1f})",
                    f"Speed: {selected_ue.speed:.1f} m/s",
                    f"Requested: {selected_ue.requested_bitrate:.1f} Mbps",
                    f"Connected BS: {station_id}",
                    f"Handover: {self.handover_reason.get(selected_ue.ue_id, 'None')}",
                ])
                if station_id is not None:
                    station = self.env.bs_list[station_id]
                    lines.extend([
                        f"Station type: {station.bs_type.upper()}",
                        f"Signal: {rsrp.get(station_id, 0):.1f} dB",
                        f"Allocated: {station.allocated_bitrate:.1f}/"
                        f"{station.total_bitrate:.1f} Mbps",
                        f"Utilization: {station.compute_rbur():.1%}",
                    ])

        if self.selected_bs is not None:
            station = self.env.bs_list[self.selected_bs]
            connected_ues = station.get_connected_users()
            lines.extend([
                "",
                f"Selected BS: {station.bs_id}",
                f"Type: {station.bs_type.upper()}",
                f"Position: ({station.position[0]:.1f}, "
                f"{station.position[1]:.1f})",
                f"Capacity: {station.total_bitrate:.1f} Mbps",
                f"Allocated: {station.allocated_bitrate:.1f} Mbps",
                f"Utilization: {station.compute_rbur():.1%}",
                f"Connected UEs: {connected_ues or 'None'}",
            ])

            if station.bs_type in {"lte", "nr"}:
                lines.extend([
                    f"Frequency: {station.carrier_frequency} MHz",
                    f"Antenna power: {station.antenna_power}",
                    f"Antenna gain: {station.antenna_gain}",
                    f"Feeder loss: {station.feeder_loss}",
                    f"PRBs: {station.allocated_prb}/"
                    f"{station.total_prb}",
                ])
                if station.bs_type == "nr":
                    lines.append(f"Numerology: {station.numerology}")
            elif station.bs_type == "sat":
                lines.extend([
                    f"Symbols: {station.frame_utilization}/"
                    f"{station.total_symbols}",
                    f"Frame utilization: "
                    f"{station.frame_utilization / station.total_symbols:.1%}",
                ])

        self.status_ax.clear()
        self.status_ax.axis("off")
        self.status_ax.text(0, 1, "\n".join(lines), va="top", fontsize=9)
        self.figure.canvas.draw_idle()


def main() -> None:
    InteractiveSimulator()
    plt.show()


if __name__ == "__main__":
    main()
