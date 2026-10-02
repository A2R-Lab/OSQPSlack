"""Building energy management problem of the paper, over one day.

Two thermal zones coupled through an open door: zone 1 faces the outside on a
warm day (and absorbs an afternoon heat event), zone 2 has the HVAC unit. A
20 kWh battery and a grid connection with time-of-use prices supply the HVAC
and a base load. Units are kW, kWh, degrees Celsius, and hours.

Decision vector: the states x_0..x_N (zone 1 and 2 temperatures, battery
charge), then the inputs u_0..u_{N-1} (HVAC power, battery discharge power,
grid import). The four soft constraint groups and their penalties:

    comfort         zone temperatures in the comfort band     quadratic
    grid contract   grid import below the contracted limit    l-inf over the day
    battery health  state of charge in [20, 80] %             elastic net
    HVAC ramp       HVAC power ramp rate below its limit      Huber
"""

import numpy as np
from scipy import linalg, sparse

from osqpslack.penalties import HUBER, L1L2, NORMINF, Penalty

DT = 0.25  # sample time [h]
N = 96  # horizon: one day
NX, NU = 3, 3
TEMP1, TEMP2, CHARGE = 0, 1, 2  # states
HVAC, BATTERY, GRID = 0, 1, 2  # inputs

# Thermal model: conductances [1/h] to the outside and through the door, and
# the temperature drop in zone 2 per kWh of HVAC electrical energy.
OUTSIDE_CONDUCTANCE = np.array([0.10, 0.12])
DOOR_CONDUCTANCE = 1.0
COOLING_EFFECT = 0.45
INITIAL_TEMPERATURE = (22.0, 22.5)

BATTERY_CAPACITY = 20.0
BATTERY_POWER_LIMIT = 5.0
INITIAL_CHARGE = 5.0
SOC_BAND = (0.2, 0.8)  # soft, fraction of capacity

HVAC_POWER_LIMIT = 8.0
FEEDER_LIMIT = 25.0  # hard, physical grid limit
CONTRACT_LIMIT = 8.0  # soft, billed by its daily peak overage
RAMP_LIMIT = 1.6  # soft, HVAC ramp rate [kW/h]

COMFORT_BAND = (21.0, 24.0)  # soft, during occupancy
SETBACK = 10.0  # the comfort band widens by this outside occupancy
SAFETY_BAND = (16.0, 32.0)  # hard
OCCUPANCY = (8.0, 20.0)

OUTSIDE_MEAN, OUTSIDE_AMPLITUDE, OUTSIDE_PEAK_HOUR = 26.0, 9.0, 15.0
BASE_LOAD = (2.0, 4.0)  # unoccupied, occupied [kW]
HEAT_GAIN = (0.10, 0.45)  # unoccupied, occupied [degC/h]
HEAT_EVENT, HEAT_EVENT_GAIN = (13.0, 18.0), 0.40  # zone 1 only
PRICES = (0.10, 0.18, 0.40)  # off-peak (before 14 h), shoulder, peak [per kWh]
PEAK = (14.0, 20.0)

POWER_WEIGHT = 1e-3  # tie-breaking regularization of all inputs
RAMP_WEIGHT = 5e-3  # tie-breaking regularization of HVAC power changes

# Penalty weights. Separable penalties sum over time steps, so they are
# scaled by DT; the l-inf penalty prices a peak and is not.
COMFORT_WEIGHT = DT * 2.0
CONTRACT_WEIGHT = 0.9
SOC_WEIGHTS = (DT * 0.2, DT * 0.2 / 0.3)  # l1, quadratic
RAMP_HUBER = (DT * 0.5, 0.5)  # weight, delta

STATE_HOURS = (np.arange(N + 1) % N) * DT  # hour of day of x_k
INPUT_HOURS = np.arange(N) * DT  # hour of day at the start of u_k


def between(hours, window):
    return (hours >= window[0]) & (hours < window[1])


def outside_temperature():
    return OUTSIDE_MEAN + OUTSIDE_AMPLITUDE * np.cos(2 * np.pi * (INPUT_HOURS - OUTSIDE_PEAK_HOUR) / 24)


def price():
    return np.select([INPUT_HOURS < PEAK[0], between(INPUT_HOURS, PEAK)], PRICES[::2], PRICES[1])


def comfort_band():
    """Comfort band at the state times; a state at 20 h closes the occupied period."""
    occupied = (STATE_HOURS >= OCCUPANCY[0]) & (STATE_HOURS <= OCCUPANCY[1])
    setback = np.where(occupied, 0.0, SETBACK)
    return COMFORT_BAND[0] - setback, COMFORT_BAND[1] + setback


def dynamics():
    """Zero-order-hold discretization x+ = Ad x + Bd u + c_k."""
    F = DOOR_CONDUCTANCE * np.array([[-1.0, 1.0], [1.0, -1.0]]) - np.diag(OUTSIDE_CONDUCTANCE)
    # Continuous inputs of the zone model: HVAC power, outside temperature,
    # and the internal heat gain of each zone.
    G = np.column_stack(([0.0, -COOLING_EFFECT], OUTSIDE_CONDUCTANCE, np.eye(2)))
    M = linalg.expm(DT * np.block([[F, G], [np.zeros((4, 6))]]))
    Ad, Gd = M[:2, :2], M[:2, 2:]

    A = np.eye(NX)
    A[:2, :2] = Ad
    B = np.zeros((NX, NU))
    B[:2, HVAC] = Gd[:, 0]
    B[CHARGE, BATTERY] = -DT

    gain = np.full((N, 2), HEAT_GAIN[0])
    gain[between(INPUT_HOURS, OCCUPANCY)] = HEAT_GAIN[1]
    gain[between(INPUT_HOURS, HEAT_EVENT), 0] += HEAT_EVENT_GAIN
    c = np.zeros((N, NX))
    c[:, :2] = np.column_stack((outside_temperature(), gain)) @ Gd[:, 1:].T
    return A, B, c


def x_index(state):
    return np.arange(N + 1) * NX + state


def u_index(inp):
    return (N + 1) * NX + np.arange(N) * NU + inp


def states(z):
    return z[: (N + 1) * NX].reshape(N + 1, NX)


def inputs(z):
    return z[(N + 1) * NX :].reshape(N, NU)


def problem():
    """The soft QP `(P, q, A, l, u, penalty)`."""
    n = (N + 1) * NX + N * NU

    def select(cols, values=1.0):
        rows = np.arange(len(cols))
        return sparse.csr_matrix((np.broadcast_to(values, len(cols)), (rows, cols)), shape=(len(cols), n))

    # Objective: energy cost of the grid import plus tie-breaking regularization.
    D = sparse.diags_array([1.0, -1.0], offsets=[0, -1], shape=(N, N))  # u_k - u_{k-1}, u_{-1} = 0
    P = sparse.block_diag((sparse.csr_matrix(((N + 1) * NX, (N + 1) * NX)), DT * POWER_WEIGHT * sparse.eye(N * NU)))
    P = P + RAMP_WEIGHT / DT * select(u_index(HVAC)).T @ (D.T @ D) @ select(u_index(HVAC))
    q = np.zeros(n)
    q[u_index(GRID)] = DT * price()

    Ad, Bd, c = dynamics()
    load = np.where(between(INPUT_HOURS, OCCUPANCY), BASE_LOAD[1], BASE_LOAD[0])
    ramp = sparse.diags_array([-1.0, 1.0], offsets=[0, 1], shape=(N - 1, N)) / DT
    lower, upper = comfort_band()
    temperatures = np.concatenate((x_index(TEMP1), x_index(TEMP2)))
    x0 = np.array([*INITIAL_TEMPERATURE, INITIAL_CHARGE])
    # x_{k+1} - Ad x_k - Bd u_k = c_k
    x_next = sparse.kron(sparse.eye(N, N + 1, k=1), sparse.eye(NX)) - sparse.kron(sparse.eye(N, N + 1), Ad)
    dyn = sparse.hstack((x_next, -sparse.kron(sparse.eye(N), Bd)))

    # (name, A rows, l, u), in row order.
    blocks = [
        ("initial", select(np.arange(NX)), x0, x0),
        ("dynamics", dyn, c.ravel(), c.ravel()),
        ("power balance", select(u_index(GRID)) + select(u_index(BATTERY)) - select(u_index(HVAC)), load, load),
        ("hvac power", select(u_index(HVAC)), 0.0, HVAC_POWER_LIMIT),
        ("battery power", select(u_index(BATTERY)), -BATTERY_POWER_LIMIT, BATTERY_POWER_LIMIT),
        ("grid feeder", select(u_index(GRID)), 0.0, FEEDER_LIMIT),
        ("battery capacity", select(x_index(CHARGE)), 0.0, BATTERY_CAPACITY),
        ("terminal charge", select(x_index(CHARGE)[-1:]), INITIAL_CHARGE, BATTERY_CAPACITY),
        ("safety", select(temperatures), SAFETY_BAND[0], SAFETY_BAND[1]),
        ("comfort", select(temperatures), np.tile(lower, 2), np.tile(upper, 2)),
        ("grid contract", select(u_index(GRID)), -np.inf, CONTRACT_LIMIT),
        ("battery health", select(x_index(CHARGE)), SOC_BAND[0] * BATTERY_CAPACITY, SOC_BAND[1] * BATTERY_CAPACITY),
        ("hvac ramp", ramp @ select(u_index(HVAC)), -RAMP_LIMIT, RAMP_LIMIT),
    ]
    A = sparse.vstack([block[1] for block in blocks], format="csc")
    l = np.concatenate([np.broadcast_to(block[2], block[1].shape[0]) for block in blocks])
    u = np.concatenate([np.broadcast_to(block[3], block[1].shape[0]) for block in blocks])

    start = np.cumsum([0] + [block[1].shape[0] for block in blocks])
    rows = {block[0]: np.arange(start[i], start[i + 1]) for i, block in enumerate(blocks)}
    penalty = Penalty.hard(A.shape[0])
    penalty.soften(rows["comfort"], L1L2, alpha2=COMFORT_WEIGHT)
    penalty.soften(rows["grid contract"], NORMINF, alpha1=CONTRACT_WEIGHT, group=0)
    penalty.soften(rows["battery health"], L1L2, alpha1=SOC_WEIGHTS[0], alpha2=SOC_WEIGHTS[1])
    penalty.soften(rows["hvac ramp"], HUBER, alpha1=RAMP_HUBER[0], delta=RAMP_HUBER[1])
    return sparse.csc_matrix(P), q, A, l, u, penalty
