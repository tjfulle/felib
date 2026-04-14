import numpy as np
import matplotlib.pyplot as plt


def sdof_central_difference(
    m: float,
    k: float,
    c: float,
    u0: float,
    v0: float,
    period: float,
    dt: float,
):
    """
    Solve m*u_ddot + c*u_dot + k*u = 0
    using a central-difference style explicit update.

    Returns
    -------
    t : np.ndarray
        Time array
    u_num : np.ndarray
        Numerical displacement history
    v_num : np.ndarray
        Numerical velocity history
    a_num : np.ndarray
        Numerical acceleration history
    """

    if m <= 0.0:
        raise ValueError("Mass m must be positive.")
    if k < 0.0:
        raise ValueError("Stiffness k must be non-negative.")
    if c < 0.0:
        raise ValueError("Damping c must be non-negative.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if period <= 0.0:
        raise ValueError("Period must be positive.")

    ninc = max(1, int(np.ceil(period / dt)))
    dt = period / ninc
    t = np.linspace(0.0, period, ninc + 1)

    u_num = np.zeros(ninc + 1, dtype=float)
    v_num = np.zeros(ninc + 1, dtype=float)
    a_num = np.zeros(ninc + 1, dtype=float)

    u_num[0] = u0
    v_num[0] = v0
    a_num[0] = (-c * v0 - k * u0) / m

    # Central difference uses velocity at half-step
    v_half = v0 - 0.5 * dt * a_num[0]

    print(f"Using fixed explicit dt = {dt:.6e} with ninc = {ninc}")

    for i in range(1, ninc + 1):
        # Acceleration from current state
        a = (-c * v_num[i - 1] - k * u_num[i - 1]) / m

        # Update half-step velocity and displacement
        v_half += dt * a
        u_num[i] = u_num[i - 1] + dt * v_half

        # Recover full-step velocity for output
        v_num[i] = v_half - 0.5 * dt * a
        a_num[i] = (-c * v_num[i] - k * u_num[i]) / m

    return t, u_num, v_num, a_num


def analytical_undamped_free_vibration(
    m: float,
    k: float,
    u0: float,
    v0: float,
    t: np.ndarray,
):
    """
    Exact solution for undamped free vibration:
        m*u_ddot + k*u = 0
    """
    wn = np.sqrt(k / m)
    u_exact = u0 * np.cos(wn * t) + (v0 / wn) * np.sin(wn * t)
    return u_exact, wn


def main():
    # SDOF parameters
    m = 1.0
    k = 100.0
    c = 0.0

    # Initial conditions
    u0 = 0.01
    v0 = 0.0

    # Time settings
    period = 2.0
    wn = np.sqrt(k / m)
    dt_crit = 2.0 / wn
    dt = 0.02  # stable since dt < dt_crit for this case

    print(f"Natural frequency wn = {wn:.6f} rad/s")
    print(f"Critical time step estimate = {dt_crit:.6e}")

    t, u_num, v_num, a_num = sdof_central_difference(
        m=m,
        k=k,
        c=c,
        u0=u0,
        v0=v0,
        period=period,
        dt=dt,
    )

    u_exact, wn = analytical_undamped_free_vibration(
        m=m,
        k=k,
        u0=u0,
        v0=v0,
        t=t,
    )

    abs_err = np.abs(u_num - u_exact)
    max_err = np.max(abs_err)
    rms_err = np.sqrt(np.mean((u_num - u_exact) ** 2))

    print(f"Max abs displacement error = {max_err:.6e}")
    print(f"RMS displacement error     = {rms_err:.6e}")

    plt.figure()
    plt.plot(t, u_exact, label="Analytical")
    plt.plot(t, u_num, "--", label="Central difference")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.title("SDOF free vibration verification")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t, abs_err)
    plt.xlabel("Time")
    plt.ylabel("Absolute error")
    plt.title("SDOF displacement error")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()