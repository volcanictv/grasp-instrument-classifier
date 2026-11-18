"""Fixed class orders the checkpoints were trained against. Do not reorder --
each list's index *is* a model's output logit index."""

CLASS_NAMES = [
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Monopolar Curved Scissors",
    "Suction Instrument",
    "Clip Applier",
    "Laparoscopic Grasper",
]

# EndoVis 2018 instrument types in the order of the EndoVis members (it has an Ultrasound Probe and no Laparoscopic Grasper).
ENDOVIS18_CLASS_NAMES = [
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Monopolar Curved Scissors",
    "Ultrasound Probe",
    "Suction Instrument",
    "Clip Applier",
]

CLASS_SETS = {"grasp": CLASS_NAMES, "endovis18": ENDOVIS18_CLASS_NAMES}
