"""Résolution de l'équation de diffusion 1D par différences finies."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParametresDiffusion:
    """Paramètres physiques et numériques du problème."""

    longueur_demi_domaine: float = 10.0
    coefficient_diffusion: float = 1.0
    nombre_intervalles: int = 100
    pas_temps: float = 0.01

    def __post_init__(self) -> None:
        if self.longueur_demi_domaine <= 0:
            raise ValueError("La demi-longueur doit être positive.")
        if self.coefficient_diffusion <= 0:
            raise ValueError("Le coefficient de diffusion doit être positif.")
        if not isinstance(self.nombre_intervalles, int) or isinstance(self.nombre_intervalles, bool) or self.nombre_intervalles < 2:
            raise ValueError("Le maillage doit contenir au moins deux intervalles.")
        if self.pas_temps <= 0:
            raise ValueError("Le pas de temps doit être positif.")
        if self.nombre_courant > 0.5:
            raise ValueError(
                "Schéma explicite instable : D*dt/dx² doit être inférieur ou égal à 0,5."
            )

    @property
    def pas_espace(self) -> float:
        return 2.0 * self.longueur_demi_domaine / self.nombre_intervalles

    @property
    def nombre_courant(self) -> float:
        return self.coefficient_diffusion * self.pas_temps / self.pas_espace**2

    @property
    def positions(self) -> np.ndarray:
        return np.linspace(
            -self.longueur_demi_domaine,
            self.longueur_demi_domaine,
            self.nombre_intervalles + 1,
        )


def creer_condition_initiale(
    positions: np.ndarray,
    valeur_gauche: float = 1.0,
    valeur_droite: float = 10.0,
) -> np.ndarray:
    """Construit une condition initiale discontinue à x=0."""
    valeurs = np.where(positions < 0.0, valeur_gauche, valeur_droite).astype(float)
    # Au point de discontinuité, la demi-somme préserve l'intégrale du profil
    # continu lors de l'intégration discrète par la règle des trapèzes.
    valeurs[positions == 0.0] = 0.5 * (valeur_gauche + valeur_droite)
    return valeurs


def avancer_un_pas_neumann(
    concentration: np.ndarray,
    nombre_courant: float,
) -> np.ndarray:
    """Avance d'un pas avec un schéma explicite et un flux nul aux frontières."""
    if not 0.0 <= nombre_courant <= 0.5:
        raise ValueError("Le nombre courant doit appartenir à l'intervalle [0 ; 0,5].")

    concentration = np.asarray(concentration, dtype=float)
    if concentration.ndim != 1 or concentration.size < 3 or not np.all(np.isfinite(concentration)):
        raise ValueError("Le champ doit contenir au moins trois valeurs finies.")
    nouvelle = np.empty_like(concentration, dtype=float)
    nouvelle[1:-1] = (
        nombre_courant * concentration[:-2]
        + (1.0 - 2.0 * nombre_courant) * concentration[1:-1]
        + nombre_courant * concentration[2:]
    )

    # Les nœuds fictifs imposent du/dx=0 avec une différence centrée.
    nouvelle[0] = (
        (1.0 - 2.0 * nombre_courant) * concentration[0]
        + 2.0 * nombre_courant * concentration[1]
    )
    nouvelle[-1] = (
        (1.0 - 2.0 * nombre_courant) * concentration[-1]
        + 2.0 * nombre_courant * concentration[-2]
    )
    return nouvelle


def integrale_trapezes(valeurs: np.ndarray, pas_espace: float) -> float:
    """Calcule l'intégrale spatiale par la règle des trapèzes."""
    return float(pas_espace * (0.5 * valeurs[0] + np.sum(valeurs[1:-1]) + 0.5 * valeurs[-1]))


def simuler_neumann_explicite(
    parametres: ParametresDiffusion,
    temps_final: float,
    instants_sauvegarde: tuple[float, ...] = (),
) -> dict[float, np.ndarray]:
    """Simule la diffusion et retourne les champs aux instants demandés."""
    if temps_final < 0:
        raise ValueError("Le temps final doit être positif ou nul.")

    positions = parametres.positions
    concentration = creer_condition_initiale(positions)
    instants = sorted(set((0.0, temps_final, *instants_sauvegarde)))
    if any(t < 0 or t > temps_final for t in instants):
        raise ValueError("Les instants de sauvegarde doivent appartenir au temps simulé.")

    indices = {}
    for t in instants:
        indice = int(round(t / parametres.pas_temps))
        if not np.isclose(indice * parametres.pas_temps, t, rtol=1e-12, atol=1e-12):
            raise ValueError("Chaque instant sauvegardé doit être un multiple du pas de temps.")
        indices[indice] = t
    nombre_pas = int(round(temps_final / parametres.pas_temps))
    if not np.isclose(nombre_pas * parametres.pas_temps, temps_final):
        raise ValueError("Le temps final doit être un multiple du pas de temps.")

    resultats = {0.0: concentration.copy()}
    for indice in range(1, nombre_pas + 1):
        concentration = avancer_un_pas_neumann(
            concentration, parametres.nombre_courant
        )
        if indice in indices:
            resultats[indices[indice]] = concentration.copy()
    return resultats
