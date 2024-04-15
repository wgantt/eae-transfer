import os

from enum import Enum, auto

##############
# CHANGE ME! #
##############
ROOT = "/brtx/603-nvme1/wgantt/eae-transfer/"

# --- #
# ACE #
# --- #
DATA_DIR = os.path.join(ROOT, "data")
ACE_DIR = os.path.join(DATA_DIR, "ace")
ACE_DATA_FILES = {
    "train": os.path.join(ACE_DIR, "train.json"),
    "dev": os.path.join(ACE_DIR, "dev.json"),
    "test": os.path.join(ACE_DIR, "test.json"),
}
ACE_ROLE_MAPPINGS = {
    # These are the roles that are actually in the ontology
    # but that can be mapped to ones that are
    "Conflict:Attack": {"Agent": "Attacker"},
    "Life:Die": {"Person": "Victim"},
}
ACE_IGNORED_ROLES = {
    # These aren't actually in the ontology, but a couple
    # of examples have them annotated
    "Movement:Transport": {"Victim", "Place"}
}
ACE_ONTOLOGY_FILE = os.path.join(ACE_DIR, "ontology", "ontology.v1.json")

# -------------------- #
# ERE (Light and Rich) #
# -------------------- #
ERE_DIR = os.path.join(DATA_DIR, "ere")
ERE_SUBTYPE_MAPPINGS = {
    "arrestjail": "Arrest-Jail",
    "beborn": "Be-Born",
    "chargeindict": "Charge-Indict",
    "declarebankruptcy": "Declare-Bankruptcy",
    "endorg": "End-Org",
    "endposition": "End-Position",
    "mergeorg": "Merge-Org",
    "releaseparole": "Release-Parole",
    "startposition": "Start-Position",
    "startorg": "Start-Org",
    "transfermoney": "Transfer-Money",
    "transferownership": "Transfer-Ownership",
    "transportartifact": "Transport-Artifact",
    "transportperson": "Transport-Person",
    "trialhearing": "Trial-Hearing",
    "tryholdhearing": "Trial-Hearing",
}
LIGHT_ERE_DIR = os.path.join(ERE_DIR, "light_ere")
LIGHT_ERE_FILES = {
    "train": os.path.join(LIGHT_ERE_DIR, "train.jsonl"),
    "dev": os.path.join(LIGHT_ERE_DIR, "dev.jsonl"),
    "test": os.path.join(LIGHT_ERE_DIR, "test.jsonl"),
}
LIGHT_ERE_ONTOLOGY_FILE = os.path.join(LIGHT_ERE_DIR, "ontology", "ontology.v2.json")
RICH_ERE_DIR = os.path.join(ERE_DIR, "rich_ere")
RICH_ERE_FILES = {
    "train": os.path.join(RICH_ERE_DIR, "train.jsonl"),
    "dev": os.path.join(RICH_ERE_DIR, "dev.jsonl"),
    "test": os.path.join(RICH_ERE_DIR, "test.jsonl"),
}
RICH_ERE_ONTOLOGY_FILE = os.path.join(RICH_ERE_DIR, "ontology", "ontology.v1.json")
ERE_NUM_CONTEXT_SENTENCES = 2

# ---------- #
# WikiEvents #
# ---------- #
WIKIEVENTS_DIR = os.path.join(DATA_DIR, "wikievents")
WIKIEVENTS_DATA_FILES = {
    "train": os.path.join(WIKIEVENTS_DIR, "train.jsonl"),
    "dev": os.path.join(WIKIEVENTS_DIR, "dev.jsonl"),
    "test": os.path.join(WIKIEVENTS_DIR, "test.jsonl"),
}
WIKIEVENTS_COREF_FILES = {
    "train": os.path.join(WIKIEVENTS_DIR, "train_coref.jsonl"),
    "dev": os.path.join(WIKIEVENTS_DIR, "dev_coref.jsonl"),
    "test": os.path.join(WIKIEVENTS_DIR, "test_coref.jsonl"),
}

WIKIEVENTS_ONTOLOGY_FILE = os.path.join(WIKIEVENTS_DIR, "ontology", "ontology.json")
# In the original WikiEvents paper (https://aclanthology.org/2021.naacl-main.69.pdf),
# examples with the following event types are dropped
WIKIEVENTS_IGNORED_TYPES = {
    "Movement.Transportation.GrantAllowPassage",
    "Transaction.AidBetweenGovernments.Unspecified",
    "Personnel.ChangePosition.Unspecified",
}
# In the original WikiEvents paper, the following event types are merged
WIKIEVENTS_MERGED_TYPES = {
    "Contact.Prevarication.Correspondence": "Contact.Prevarication.Unspecified",
    "Contact.Prevarication.Meet": "Contact.Prevarication.Unspecified",
    "Contact.RequestCommand.Broadcast": "Contact.RequestCommand.Unspecified",
    "Contact.RequestCommand.Correspondence": "Contact.RequestCommand.Unspecified",
    "Contact.RequestCommand.Meet": "Contact.RequestCommand.Unspecified",
    "Contact.ThreatenCoerce.Broadcast": "Contact.ThreatenCoerce.Unspecified",
    "Contact.ThreatenCoerce.Correspondence": "Contact.ThreatenCoerce.Unspecified",
    "Contact.ThreatenCoerce.Meet": "Contact.ThreatenCoerce.Unspecified",
}
# Based on the context length used in the original paper
WIKIEVENTS_MAX_CONTEXT_LEN = 400

# ---- #
# RAMS #
# ---- #
RAMS_DIR = os.path.join(DATA_DIR, "rams")
RAMS_DATA_FILES = {
    "train": os.path.join(RAMS_DIR, "train.jsonl"),
    "dev": os.path.join(RAMS_DIR, "dev.jsonl"),
    "test": os.path.join(RAMS_DIR, "test.jsonl"),
}
RAMS_ONTOLOGY_FILE = os.path.join(RAMS_DIR, "ontology", "ontology.json")
RAMS_ROLE_TO_UPPER = {
    "artifact": "Artifact",
    "artifactmoney": "ArtifactMoney",
    "attacker": "Attacker",
    "ballot": "Ballot",
    "beneficiary": "Beneficiary",
    "candidate": "Candidate",
    "communicator": "Communicator",
    "crashobject": "CrashObject",
    "crime": "Crime",
    "damager": "Damager",
    "destroyer": "Destroyer",
    "damagerdestroyer": "DamagerDestroyer",
    "deceased": "Deceased",
    "defendant": "Defendant",
    "demonstrator": "Demonstrator",
    "destination": "Destination",
    "detainee": "Detainee",
    "driverpassenger": "DriverPassenger",
    "employee": "Employee",
    "executioner": "Executioner",
    "extraditer": "Extraditer",
    "fireexplosionobject": "FireExplosionObject",
    "founder": "Founder",
    "giver": "Giver",
    "governmentbody": "GovernmentBody",
    "gpe": "GPE",
    "granter": "Granter",
    "hidingplace": "HidingPlace",
    "injurer": "Injurer",
    "inspectedentity": "InspectedEntity",
    "inspector": "Inspector",
    "instrument": "Instrument",
    "investigator": "Investigator",
    "jailer": "Jailer",
    "judgecourt": "JudgeCourt",
    "killer": "Killer",
    "law": "Law",
    "manufacturer": "Manufacturer",
    "money": "Money",
    "monitor": "Monitor",
    "monitoredentity": "MonitoredEntity",
    "observedentity": "ObservedEntity",
    "observer": "Observer",
    "origin": "Origin",
    "otherparticipant": "OtherParticipant",
    "participant": "Participant",
    "passenger": "Passenger",
    "place": "Place",
    "placeofemployment": "PlaceOfEmployment",
    "preventer": "Preventer",
    "prosecutor": "Prosecutor",
    "recipient": "Recipient",
    "rejecternullifier": "RejecterNullifier",
    "result": "Result",
    "retreater": "Retreater",
    "spy": "Spy",
    "surrenderer": "Surrenderer",
    "target": "Target",
    "territoryorfacility": "TerritoryOrFacility",
    "transporter": "Transporter",
    "vehicle": "Vehicle",
    "victim": "Victim",
    "violator": "Violator",
    "voter": "Voter",
    "yielder": "Yielder",
}

# -------- #
# FRAMENET #
# -------- #
FRAMENET_DIR = os.path.join(DATA_DIR, "framenet")
FRAMENET_DATA_FILES = {
    "train": os.path.join(FRAMENET_DIR, "train.jsonl"),
    "dev": os.path.join(FRAMENET_DIR, "dev.jsonl"),
    "test": os.path.join(FRAMENET_DIR, "test.jsonl"),
}
FRAMENET_ONTOLOGY_FILE = os.path.join(FRAMENET_DIR, "ontology", "ontology.v1.json")

# ----- #
# FAMUS #
# ----- #
FAMUS_DIR = os.path.join(DATA_DIR, "famus")
FAMUS_DATA_FILES = {
    "train": os.path.join(FAMUS_DIR, "train.jsonl"),
    "dev": os.path.join(FAMUS_DIR, "dev.jsonl"),
    "test": os.path.join(FAMUS_DIR, "test.jsonl"),
}
FAMUS_ONTOLOGY_FILE = os.path.join(FAMUS_DIR, "ontology", "ontology.json")


class DatasetChoice(Enum):
    ACE = auto()
    ERE_LIGHT = auto()
    ERE_RICH = auto()
    FAMUS_REPORTS = auto()
    FRAMENET = auto()
    FRAMENET_CORE_ROLES = auto()
    FRAMENET_FOR_FAMUS = auto()
    RAMS = auto()
    WIKIEVENTS = auto()


class TaskChoice(Enum):
    EAE = auto()
    OTE = auto()
    QA = auto()
    INFILLING = auto()


# delineates distinct events
EVENT_SEP = "<event_sep>"

# delineats distinct sets of role fillers within the same event
ROLE_SEP = "<role_sep>"
ARG_SEP = "<arg_sep>"
TOKENS_TO_ADD = [EVENT_SEP, ROLE_SEP, ARG_SEP]


def _validate_split(split: str) -> None:
    assert split in {"train", "dev", "test"}


def _validate_mode(mode: str) -> None:
    assert mode in {"light_ere", "rich_ere"}
