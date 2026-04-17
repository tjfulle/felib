import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def buildBarMesh(length, numElems):
    nodes = np.linspace(0.0, length, numElems + 1)
    conn = [(i, i + 1) for i in range(numElems)]
    return nodes, conn


def barElementStiffness(E, area, le):
    return (E * area / le) * np.array([
        [1.0, -1.0],
        [-1.0,  1.0]
    ])


def barElementLumpedMass(rho, area, le):
    return (rho * area * le / 2.0) * np.array([1.0, 1.0])


def assembleSystem(nodes, conn, E, rho, area):
    numNodes = len(nodes)
    K = np.zeros((numNodes, numNodes))
    M = np.zeros(numNodes)

    for n1, n2 in conn:
        le = nodes[n2] - nodes[n1]
        ke = barElementStiffness(E, area, le)
        me = barElementLumpedMass(rho, area, le)

        dofs = [n1, n2]

        for a in range(2):
            M[dofs[a]] += me[a]
            for b in range(2):
                K[dofs[a], dofs[b]] += ke[a, b]

    return K, M


def computeElementStress(nodes, conn, u, E):
    stress = np.zeros(len(conn))
    for e, (n1, n2) in enumerate(conn):
        le = nodes[n2] - nodes[n1]
        strain = (u[n2] - u[n1]) / le
        stress[e] = E * strain
    return stress


def prescribedVelocity(t, pulseDuration, v0):
    if 0.0 <= t <= pulseDuration:
        return v0 * np.sin(np.pi * t / pulseDuration)
    return 0.0


def prescribedDisplacement(t, pulseDuration, v0):
    if t <= 0.0:
        return 0.0
    if t <= pulseDuration:
        return (v0 * pulseDuration / np.pi) * (1.0 - np.cos(np.pi * t / pulseDuration))
    return 2.0 * v0 * pulseDuration / np.pi


def analyticalStressIncident(x, t, rho, c, pulseDuration, v0):
    tau = t - x / c
    if 0.0 <= tau <= pulseDuration:
        return -rho * c * v0 * np.sin(np.pi * tau / pulseDuration)
    return 0.0


def analyticalStressReflected(x, t, length, rho, c, pulseDuration, v0):
    tau = t - (2.0 * length - x) / c
    if 0.0 <= tau <= pulseDuration:
        return -rho * c * v0 * np.sin(np.pi * tau / pulseDuration)
    return 0.0


def analyticalStressTotal(xArray, t, length, rho, c, pulseDuration, v0):
    sigma = np.zeros_like(xArray)
    for i, x in enumerate(xArray):
        sigma[i] = (
            analyticalStressIncident(x, t, rho, c, pulseDuration, v0)
            + analyticalStressReflected(x, t, length, rho, c, pulseDuration, v0)
        )
    return sigma


def firstArrivalTimeFromSignal(timeHist, signal, thresholdFrac=0.05):
    peak = np.max(np.abs(signal))
    if peak <= 0.0:
        return None

    threshold = thresholdFrac * peak
    idx = np.where(np.abs(signal) >= threshold)[0]
    if len(idx) == 0:
        return None
    return timeHist[idx[0]]


def reflectedArrivalTimeFromSignal(timeHist, signal, theoreticalReflectionTime, thresholdFrac=0.05):
    peak = np.max(np.abs(signal))
    if peak <= 0.0:
        return None

    threshold = thresholdFrac * peak
    afterMask = timeHist >= 0.8 * theoreticalReflectionTime
    idx = np.where(afterMask & (np.abs(signal) >= threshold))[0]
    if len(idx) == 0:
        return None
    return timeHist[idx[0]]


def main():
    # -----------------------------
    # Problem setup
    # -----------------------------
    length = 1.0
    area = 1.0
    E = 200e9
    rho = 7800.0
    numElems = 100

    totalTime = 4.0e-4
    v0 = 1.0e-3
    pulseDuration = 2.0e-5

    dispScale = 5.0e5
    showAnimation = True

    # -----------------------------
    # Mesh and assembly
    # -----------------------------
    nodes, conn = buildBarMesh(length, numElems)
    K, M = assembleSystem(nodes, conn, E, rho, area)

    c = np.sqrt(E / rho)
    dx = nodes[1] - nodes[0]
    dtCritEstimate = dx / c
    dt = 0.8 * dtCritEstimate
    numSteps = int(np.ceil(totalTime / dt))
    dt = totalTime / numSteps

    print(f"Wave speed c = {c:.6e} m/s")
    print(f"Element size dx = {dx:.6e} m")
    print(f"Critical dt estimate dx/c = {dtCritEstimate:.6e} s")
    print(f"Using fixed explicit dt = {dt:.6e} s")
    print(f"numSteps = {numSteps}")

    totalMass = np.sum(M)
    exactMass = rho * area * length
    print(f"Total lumped mass = {totalMass:.6e}")
    print(f"Exact bar mass    = {exactMass:.6e}")
    print(f"Mass error        = {abs(totalMass - exactMass):.6e}")

    # -----------------------------
    # State variables
    # -----------------------------
    numNodes = len(nodes)
    leftNode = 0
    rightNode = numNodes - 1
    freeDofs = np.arange(1, rightNode)

    u = np.zeros(numNodes)
    v = np.zeros(numNodes)
    a = np.zeros(numNodes)
    vHalf = np.zeros(numNodes)

    # -----------------------------
    # Initial conditions
    # -----------------------------
    fInt = K @ u
    a[freeDofs] = -fInt[freeDofs] / M[freeDofs]
    vHalf[freeDofs] = v[freeDofs] - 0.5 * dt * a[freeDofs]

    u[leftNode] = prescribedDisplacement(0.0, pulseDuration, v0)
    u[rightNode] = 0.0
    v[rightNode] = 0.0
    a[rightNode] = 0.0
    vHalf[rightNode] = 0.0

    # -----------------------------
    # Storage
    # -----------------------------
    trackedElems = [5, numElems // 2, numElems - 6]
    stressHist = {e: [] for e in trackedElems}
    analyticalStressHist = {e: [] for e in trackedElems}

    timeHist = []
    stressFieldHistory = []
    uHistory = []

    elemMidpoints = np.zeros(len(conn))
    for e, (n1, n2) in enumerate(conn):
        elemMidpoints[e] = 0.5 * (nodes[n1] + nodes[n2])

    sigmaExpected = rho * c * v0
    print(f"Expected incident stress magnitude scale = {sigmaExpected:.6e} Pa")

    # -----------------------------
    # Time integration loop
    # -----------------------------
    for step in range(numSteps):
        tNew = (step + 1) * dt

        fInt = K @ u

        a[freeDofs] = -fInt[freeDofs] / M[freeDofs]
        vHalf[freeDofs] += dt * a[freeDofs]
        u[freeDofs] += dt * vHalf[freeDofs]

        u[leftNode] = prescribedDisplacement(tNew, pulseDuration, v0)
        u[rightNode] = 0.0
        vHalf[rightNode] = 0.0

        v[freeDofs] = vHalf[freeDofs] - 0.5 * dt * a[freeDofs]
        v[leftNode] = prescribedVelocity(tNew, pulseDuration, v0)
        v[rightNode] = 0.0

        a[leftNode] = 0.0
        a[rightNode] = 0.0

        stress = computeElementStress(nodes, conn, u, E)
        stressAnalytical = analyticalStressTotal(elemMidpoints, tNew, length, rho, c, pulseDuration, v0)

        timeHist.append(tNew)
        stressFieldHistory.append(stress.copy())
        uHistory.append(u.copy())

        for e in trackedElems:
            stressHist[e].append(stress[e])
            analyticalStressHist[e].append(stressAnalytical[e])

    # -----------------------------
    # Convert histories
    # -----------------------------
    timeHist = np.array(timeHist)
    stressFieldHistory = np.array(stressFieldHistory)
    uHistory = np.array(uHistory)

    analyticalHistory = np.array([
        analyticalStressTotal(elemMidpoints, t, length, rho, c, pulseDuration, v0)
        for t in timeHist
    ])

    # -----------------------------
    # Quantitative verification metrics
    # -----------------------------
    stressError = stressFieldHistory - analyticalHistory
    maxAbsStressError = np.max(np.abs(stressError))
    rmseStress = np.sqrt(np.mean(stressError**2))
    peakAnalyticalStress = np.max(np.abs(analyticalHistory))
    nrmseStress = rmseStress / peakAnalyticalStress if peakAnalyticalStress > 0.0 else np.nan

    print("\nFull-field stress comparison")
    print(f"  Max absolute stress error = {maxAbsStressError:.6e} Pa")
    print(f"  RMSE stress error         = {rmseStress:.6e} Pa")
    print(f"  Normalized RMSE           = {nrmseStress:.6%}")

    print("\nTracked element arrival-time checks")
    for e in trackedElems:
        xMid = elemMidpoints[e]
        theoreticalIncident = xMid / c
        theoreticalReflection = (2.0 * length - xMid) / c

        sigNum = np.array(stressHist[e])

        numericalIncident = firstArrivalTimeFromSignal(timeHist, sigNum, thresholdFrac=0.05)
        numericalReflection = reflectedArrivalTimeFromSignal(
            timeHist, sigNum, theoreticalReflectionTime=theoreticalReflection, thresholdFrac=0.05
        )

        incidentError = None if numericalIncident is None else numericalIncident - theoreticalIncident
        reflectionError = None if numericalReflection is None else numericalReflection - theoreticalReflection

        print(f"  Element {e:3d} at x = {xMid:.6f} m")
        print(f"    Incident:   theory = {theoreticalIncident:.6e} s, "
              f"numerical = {numericalIncident:.6e} s, "
              f"error = {incidentError:.6e} s")
        print(f"    Reflection: theory = {theoreticalReflection:.6e} s, "
              f"numerical = {numericalReflection:.6e} s, "
              f"error = {reflectionError:.6e} s")

    leftInteriorElem = trackedElems[0]
    numericalPeak = np.max(np.abs(stressHist[leftInteriorElem]))
    peakStressErrorPct = 100.0 * abs(numericalPeak - sigmaExpected) / sigmaExpected

    print("\nPeak incident stress check")
    print(f"  Theoretical scale rho*c*v0 = {sigmaExpected:.6e} Pa")
    print(f"  Numerical peak near loaded end = {numericalPeak:.6e} Pa")
    print(f"  Percent error = {peakStressErrorPct:.3f}%")

    # -----------------------------
    # Plot: tracked element stress histories
    # -----------------------------
    plt.figure(figsize=(10, 6))
    for e in trackedElems:
        plt.plot(timeHist, stressHist[e], label=f"Numerical elem {e}")
        plt.plot(timeHist, analyticalStressHist[e], "--", label=f"Analytical elem {e}")

    plt.xlabel("Time (s)")
    plt.ylabel("Axial stress (Pa)")
    plt.title("Tracked element stress histories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Animation: stress + deformed bar
    # -----------------------------
    if showAnimation:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]}
        )

        numLine, = ax1.plot([], [], lw=2, label="Numerical")
        anaLine, = ax1.plot([], [], "--", lw=2, label="Analytical")
        ax1.axhline(0.0, color="k", linewidth=0.8)
        incMarker = ax1.axvline(0.0, linestyle="--", linewidth=1.2, label="Incident front")
        refMarker = ax1.axvline(length, linestyle=":", linewidth=1.2, label="Reflected front")
        timeText = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, va="top")

        combinedMin = min(np.min(stressFieldHistory), np.min(analyticalHistory))
        combinedMax = max(np.max(stressFieldHistory), np.max(analyticalHistory))
        yPad = 0.10 * max(abs(combinedMin), abs(combinedMax))

        ax1.set_xlim(elemMidpoints[0], elemMidpoints[-1])
        ax1.set_ylim(combinedMin - yPad, combinedMax + yPad)
        ax1.set_xlabel("Position along bar (m)")
        ax1.set_ylabel("Axial stress (Pa)")
        ax1.set_title("Stress wave propagation in bar")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        yTop1 = ax1.get_ylim()[1]
        ax1.text(elemMidpoints[0], 0.92 * yTop1, "Loaded end", ha="left")
        ax1.text(elemMidpoints[-1], 0.92 * yTop1, "Fixed wall", ha="right")

        ax2.plot(nodes, np.zeros_like(nodes), "--", lw=1.0, label="Undeformed")
        deformedLine, = ax2.plot([], [], lw=2, marker="o", markersize=3, label="Deformed")
        ax2.axhline(0.0, color="k", linewidth=0.8)

        maxDisp = np.max(np.abs(uHistory))
        yDisp = max(1.2, 1.25 * dispScale * maxDisp)

        ax2.set_xlim(0.0, length)
        ax2.set_ylim(-yDisp, yDisp)
        ax2.set_xlabel("Position along bar (m)")
        ax2.set_ylabel("Scaled transverse display")
        ax2.set_title(f"Deformed bar view (horizontal deformation scaled by {dispScale:.1e})")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        yTop2 = ax2.get_ylim()[1]
        ax2.text(0.0, 0.85 * yTop2, "Loaded end", ha="left")
        ax2.text(length, 0.85 * yTop2, "Fixed wall", ha="right")

        def incidentFrontPosition(t):
            return min(c * t, length)

        def reflectedFrontPosition(t):
            if t < length / c:
                return length
            return max(2.0 * length - c * t, 0.0)

        def init():
            numLine.set_data([], [])
            anaLine.set_data([], [])
            deformedLine.set_data([], [])
            incMarker.set_xdata([0.0, 0.0])
            refMarker.set_xdata([length, length])
            timeText.set_text("")
            return numLine, anaLine, deformedLine, incMarker, refMarker, timeText

        def update(frame):
            t = timeHist[frame]

            numLine.set_data(elemMidpoints, stressFieldHistory[frame])
            anaLine.set_data(elemMidpoints, analyticalHistory[frame])

            xInc = incidentFrontPosition(t)
            xRef = reflectedFrontPosition(t)
            incMarker.set_xdata([xInc, xInc])
            refMarker.set_xdata([xRef, xRef])

            xDef = nodes + dispScale * uHistory[frame]
            yDef = np.zeros_like(nodes)
            deformedLine.set_data(xDef, yDef)

            timeText.set_text(f"time = {t:.6e} s")
            return numLine, anaLine, deformedLine, incMarker, refMarker, timeText

        frameSkip = 2
        frames = range(0, len(timeHist), frameSkip)

        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            interval=45,
            blit=True,
            repeat=True
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()