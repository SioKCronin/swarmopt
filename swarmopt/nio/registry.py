"""Name-to-class registry for the nature-inspired corpus."""

from __future__ import annotations

from typing import Dict, Type

from .abc import ArtificialBeeColony
from .acor import AntColonyOptimization
from .ais import ClonalSelection
from .altruistic import AltruisticPopulationAlgorithm
from .animal_migration import AnimalMigrationOptimization
from .artificial_algae import ArtificialAlgaeAlgorithm
from .artificial_chemical import ArtificialChemicalReactionOptimization
from .artificial_ecosystem import ArtificialEcosystemAlgorithm
from .bacteria_chemotaxis import BacterialChemotaxis
from .bacterial_colony import BacterialColonyOptimization
from .bacterial_evolutionary import BacterialEvolutionaryAlgorithm
from .bacterial_foraging import BacterialForagingOptimization
from .bat import BatAlgorithm
from .biogeography import BiogeographyBasedOptimization
from .bird_mating import BirdMatingOptimizer
from .black_hole import BlackHoleAlgorithm
from .bull import BullOptimizationAlgorithm
from .collective_animal import CollectiveAnimalBehavior
from .cuckoo import CuckooSearch
from .cultural import CulturalAlgorithm
from .cuttlefish import CuttlefishAlgorithm
from .differential_evolution import DifferentialEvolution
from .dragonfly import DragonflyAlgorithm
from .elephant import ElephantHerdingOptimization
from .firefly import FireflyAlgorithm
from .fireworks import FireworksAlgorithm
from .fish_school import ArtificialFishSchool
from .flower_pollination import FlowerPollinationAlgorithm
from .forest import ForestOptimizationAlgorithm
from .gases_brownian import GasesBrownianMotionOptimization
from .genetic import GeneticAlgorithm
from .grey_wolf import GreyWolfOptimizer
from .gsa import GravitationalSearchAlgorithm
from .gso import GlowwormSwarmOptimization
from .hbmo import HoneyBeeMatingOptimization
from .honey_bee_marriage import MarriageInHoneyBees
from .invasive_weed import InvasiveWeedOptimization
from .iwd_co import IWDCO
from .krill import KrillHerd
from .lion import LionOptimizationAlgorithm
from .memetic import MemeticAlgorithm
from .mine_blast import MineBlastAlgorithm
from .optics import OpticsInspiredOptimization
from .philippine_eagle import PhilippineEagleOptimization
from .plant_propagation import PlantPropagationAlgorithm
from .ppso import PPSO
from .pso import ParticleSwarmOptimization
from .raven import RavenRoostingOptimization
from .roach import RoachInfestationOptimization
from .sco import SocialCognitiveOptimization
from .shuffled_frog import ShuffledFrogLeaping
from .social_spider import SocialSpiderOptimization
from .spider_monkey import SpiderMonkeyOptimization
from .spiral import SpiralDynamicAlgorithm
from .strawberry import StrawberryAlgorithm
from .water_cycle import WaterCycleAlgorithm
from .water_wave import WaterWaveOptimization

OPTIMIZERS: Dict[str, Type] = {
    "ga": GeneticAlgorithm,
    "pso": ParticleSwarmOptimization,
    "ais": ClonalSelection,
    "ma": MemeticAlgorithm,
    "acor": AntColonyOptimization,
    "ca": CulturalAlgorithm,
    "de": DifferentialEvolution,
    "bfo": BacterialForagingOptimization,
    "mhb": MarriageInHoneyBees,
    "afs": ArtificialFishSchool,
    "bc": BacterialChemotaxis,
    "sco": SocialCognitiveOptimization,
    "abc": ArtificialBeeColony,
    "gso": GlowwormSwarmOptimization,
    "hbmo": HoneyBeeMatingOptimization,
    "iwo": InvasiveWeedOptimization,
    "sfla": ShuffledFrogLeaping,
    "iwdco": IWDCO,
    "bbo": BiogeographyBasedOptimization,
    "rio": RoachInfestationOptimization,
    "bea": BacterialEvolutionaryAlgorithm,
    "cs": CuckooSearch,
    "fa": FireflyAlgorithm,
    "gsa": GravitationalSearchAlgorithm,
    "bat": BatAlgorithm,
    "peo": PhilippineEagleOptimization,
    "fwa": FireworksAlgorithm,
    "apa": AltruisticPopulationAlgorithm,
    "sda": SpiralDynamicAlgorithm,
    "strawberry": StrawberryAlgorithm,
    "aaa": ArtificialAlgaeAlgorithm,
    "bco": BacterialColonyOptimization,
    "fpa": FlowerPollinationAlgorithm,
    "kh": KrillHerd,
    "wca": WaterCycleAlgorithm,
    "ppso": PPSO,
    "da": DragonflyAlgorithm,
    "bha": BlackHoleAlgorithm,
    "cfa": CuttlefishAlgorithm,
    "gbmo": GasesBrownianMotionOptimization,
    "mba": MineBlastAlgorithm,
    "ppa": PlantPropagationAlgorithm,
    "sso": SocialSpiderOptimization,
    "smo": SpiderMonkeyOptimization,
    "amo": AnimalMigrationOptimization,
    "aea": ArtificialEcosystemAlgorithm,
    "bmo": BirdMatingOptimizer,
    "foa": ForestOptimizationAlgorithm,
    "gwo": GreyWolfOptimizer,
    "loa": LionOptimizationAlgorithm,
    "oio": OpticsInspiredOptimization,
    "rroa": RavenRoostingOptimization,
    "wwo": WaterWaveOptimization,
    "cab": CollectiveAnimalBehavior,
    "acroa": ArtificialChemicalReactionOptimization,
    "bull": BullOptimizationAlgorithm,
    "eho": ElephantHerdingOptimization,
}


def get_optimizer(name: str) -> Type:
    """Return the optimizer class registered under ``name``."""
    key = name.strip().lower()
    try:
        return OPTIMIZERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(OPTIMIZERS))
        raise KeyError(f"unknown optimizer {name!r}. Known names: {known}") from exc


#: Where each method's metaphor comes from. ``swarm``: collective animal or
#: social behaviour. ``evolutionary``: selection, mating, inheritance or
#: immunity. ``ecological``: plants, seeding and ecosystems. ``physical``:
#: physics, chemistry or optics (cuttlefish is here because the method models
#: light reflection in skin cells, not collective behaviour).
#:
#: The family names the metaphor, not the mechanism. Many metaphor-based
#: methods reduce to familiar operators (weighted moves toward good points,
#: crossover, random restarts); see Sörensen (2015), "Metaheuristics—the
#: metaphor exposed", and compare methods at an equal evaluation budget with
#: ``swarmopt.nio.compare``.
FAMILIES: Dict[str, str] = {
    **dict.fromkeys(
        ["abc", "acor", "afs", "amo", "bat", "bc", "bco", "bfo", "cab", "cs",
         "da", "eho", "fa", "gso", "gwo", "kh", "loa", "peo", "ppso", "pso",
         "rio", "rroa", "sco", "sfla", "smo", "sso"], "swarm"),
    **dict.fromkeys(
        ["ais", "apa", "bbo", "bea", "bmo", "bull", "ca", "de", "ga", "hbmo",
         "ma", "mhb"], "evolutionary"),
    **dict.fromkeys(
        ["aaa", "aea", "foa", "fpa", "iwo", "ppa", "strawberry"], "ecological"),
    **dict.fromkeys(
        ["acroa", "bha", "cfa", "fwa", "gbmo", "gsa", "iwdco", "mba", "oio",
         "sda", "wca", "wwo"], "physical"),
}

FAMILY_NAMES = ("swarm", "evolutionary", "ecological", "physical")


def family(name: str) -> str:
    """The metaphor family of a registered optimizer."""
    get_optimizer(name)
    return FAMILIES[name.strip().lower()]


def by_family(name: str) -> Dict[str, Type]:
    """Registered optimizers in one family, by short name."""
    if name not in FAMILY_NAMES:
        raise ValueError(f"family must be one of {FAMILY_NAMES}, not {name!r}")
    return {k: OPTIMIZERS[k] for k in sorted(OPTIMIZERS) if FAMILIES[k] == name}
