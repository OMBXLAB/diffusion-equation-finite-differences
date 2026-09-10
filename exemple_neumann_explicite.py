"""Exemple validé : schéma explicite avec conditions de Neumann homogènes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from diffusion import ParametresDiffusion, integrale_trapezes, simuler_neumann_explicite


def main() -> None:
    parametres = ParametresDiffusion(pas_temps=0.01)
    resultats = simuler_neumann_explicite(
        parametres,
        temps_final=200.0,
        instants_sauvegarde=(1.0, 5.0, 20.0),
    )

    positions = parametres.positions
    masse_initiale = integrale_trapezes(resultats[0.0], parametres.pas_espace)
    masse_finale = integrale_trapezes(resultats[200.0], parametres.pas_espace)
    moyenne_attendue = masse_initiale / (2.0 * parametres.longueur_demi_domaine)

    print(f"Δx = {parametres.pas_espace:.4f}")
    print(f"Δt = {parametres.pas_temps:.4f}")
    print(f"M  = {parametres.nombre_courant:.4f}")
    print(f"Intégrale initiale = {masse_initiale:.8f}")
    print(f"Intégrale finale   = {masse_finale:.8f}")
    print(f"État uniforme attendu = {moyenne_attendue:.6f}")

    dossier_resultats = Path("resultats")
    dossier_resultats.mkdir(exist_ok=True)

    figure, axe = plt.subplots(figsize=(8, 5))
    for instant, concentration in resultats.items():
        axe.plot(positions, concentration, label=f"t = {instant:g} s")
    axe.set(
        xlabel="Position x",
        ylabel="Concentration u(x,t)",
        title="Diffusion 1D - schéma explicite, flux nul aux frontières",
    )
    axe.grid(alpha=0.3)
    axe.legend()
    figure.tight_layout()
    figure.savefig(dossier_resultats / "profils_concentration.png", dpi=180)


if __name__ == "__main__":
    main()

