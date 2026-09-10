"""Comparaison indépendante avec la série de Fourier du problème continu."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from diffusion import ParametresDiffusion, simuler_neumann_explicite


def reference_fourier(x, t, longueur=10.0, diffusion=1.0, termes=200):
    """Solution pour t>0, flux nul et concentrations initiales 1 et 10."""
    if t <= 0:
        raise ValueError('La référence est évaluée pour t > 0.')
    impairs = 2 * np.arange(termes) + 1
    frequences = impairs * np.pi / (2 * longueur)
    amplitudes = 18 / (np.pi * impairs)
    return 5.5 + np.sum(
        (amplitudes * np.exp(-diffusion * frequences**2 * t))[:, None]
        * np.sin(frequences[:, None] * np.asarray(x)), axis=0)


def main():
    p = ParametresDiffusion()
    champs = simuler_neumann_explicite(p, 200.0, (1.0, 5.0, 20.0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for t in (1.0, 5.0, 20.0, 200.0):
        ref = reference_fourier(p.positions, t)
        erreur = champs[t] - ref
        print(f't={t:g} : erreur maximale={np.max(np.abs(erreur)):.8g}')
        axes[0].plot(p.positions, champs[t], label=f'Python, t={t:g}')
        axes[0].plot(p.positions[::5], ref[::5], 'k.', markersize=3)
        axes[1].plot(p.positions, erreur, label=f't={t:g}')
    axes[0].set_title('Points noirs : référence de Fourier')
    axes[0].set_ylabel('Concentration (unités arbitraires)')
    axes[1].set_title('Erreur numérique - référence')
    for axe in axes:
        axe.set_xlabel('Position x (unités de longueur)')
        axe.grid(alpha=0.3)
        axe.legend(fontsize=8)
    fig.tight_layout()
    Path('resultats').mkdir(exist_ok=True)
    fig.savefig('resultats/comparaison_fourier.png', dpi=160)


if __name__ == '__main__':
    main()
