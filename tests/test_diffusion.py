import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from diffusion import (
    ParametresDiffusion,
    avancer_un_pas_neumann,
    integrale_trapezes,
    simuler_neumann_explicite,
)


class TestsSchemaExpliciteNeumann(unittest.TestCase):
    def test_instants_non_alignes_refuses(self):
        with self.assertRaises(ValueError):
            simuler_neumann_explicite(ParametresDiffusion(), 1.0, (0.015,))

    def test_mode_analytique_neumann(self):
        p = ParametresDiffusion()
        x = p.positions
        champ = 5.5 + np.sin(np.pi * x / 20)
        for _ in range(1000):
            champ = avancer_un_pas_neumann(champ, p.nombre_courant)
        reference = 5.5 + np.sin(np.pi * x / 20) * np.exp(-(np.pi / 20)**2 * 10)
        self.assertLess(np.max(np.abs(champ - reference)), 1e-5)

    def test_creneau_reference_fourier(self):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from verifier_solution import reference_fourier
        p = ParametresDiffusion()
        champ = simuler_neumann_explicite(p, 1.0)[1.0]
        reference = reference_fourier(p.positions, 1.0)
        # Tolérance de 0,1 % de l'amplitude initiale (9 unités).
        erreur = np.max(np.abs(champ - reference))
        self.assertLess(erreur / 9.0, 0.001)
        fin = ParametresDiffusion(nombre_intervalles=200, pas_temps=0.0025)
        champ_fin = simuler_neumann_explicite(fin, 1.0)[1.0]
        erreur_fin = np.max(np.abs(champ_fin - reference_fourier(fin.positions, 1.0)))
        self.assertLess(erreur_fin, erreur / 3.5)

    def test_refus_pas_temps_instable(self) -> None:
        with self.assertRaises(ValueError):
            ParametresDiffusion(pas_temps=0.021)

    def test_champ_uniforme_inchange(self) -> None:
        champ = np.full(21, 4.2)
        suivant = avancer_un_pas_neumann(champ, nombre_courant=0.25)
        self.assertTrue(np.allclose(champ, suivant))

    def test_conservation_integrale(self) -> None:
        parametres = ParametresDiffusion(nombre_intervalles=40, pas_temps=0.05)
        resultats = simuler_neumann_explicite(parametres, 20.0)
        initiale = integrale_trapezes(resultats[0.0], parametres.pas_espace)
        finale = integrale_trapezes(resultats[20.0], parametres.pas_espace)
        self.assertAlmostEqual(initiale, 110.0, places=10)
        self.assertAlmostEqual(initiale, finale, places=10)

    def test_tendance_vers_etat_uniforme(self) -> None:
        parametres = ParametresDiffusion(nombre_intervalles=40, pas_temps=0.05)
        resultats = simuler_neumann_explicite(parametres, 200.0)
        ecart_final = np.ptp(resultats[200.0])
        self.assertLess(ecart_final, 0.1)


if __name__ == "__main__":
    unittest.main()
