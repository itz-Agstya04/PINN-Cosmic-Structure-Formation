import matplotlib.pyplot as plt
from Simulations.pic import PIC1D


def main():
    # Initialize PIC solver
    pic = PIC1D(Np=5000, Nx=64, dt=0.05)

    # Run simulation
    steps = 100
    for _ in range(steps):
        pic.step()

    # Plot phase space
    plt.figure(figsize=(6, 4))
    plt.scatter(pic.x, pic.v, s=1, alpha=0.7)
    plt.xlabel("Position x")
    plt.ylabel("Velocity v")
    plt.title("Phase Space (PIC Simulation)")
    plt.tight_layout()

    # Save and show
    plt.savefig("results/pic_phase_space.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
