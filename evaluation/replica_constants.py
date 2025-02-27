REPLICA_CLASSES = [
    "other",
    "backpack",
    "base-cabinet",
    "basket",
    "bathtub",
    "beam",
    "beanbag",
    "bed",
    "bench",
    "bike",
    "bin",
    "blanket",
    "blinds",
    "book",
    "bottle",
    "box",
    "bowl",
    "camera",
    "cabinet",
    "candle",
    "chair",
    "chopping-board",
    "clock",
    "cloth",
    "clothing",
    "coaster",
    "comforter",
    "computer-keyboard",
    "cup",
    "cushion",
    "curtain",
    "ceiling",
    "cooktop",
    "countertop",
    "desk",
    "desk-organizer",
    "desktop-computer",
    "door",
    "exercise-ball",
    "faucet",
    "floor",
    "handbag",
    "hair-dryer",
    "handrail",
    "indoor-plant",
    "knife-block",
    "kitchen-utensil",
    "lamp",
    "laptop",
    "major-appliance",
    "mat",
    "microwave",
    "monitor",
    "mouse",
    "nightstand",
    "pan",
    "panel",
    "paper-towel",
    "phone",
    "picture",
    "pillar",
    "pillow",
    "pipe",
    "plant-stand",
    "plate",
    "pot",
    "rack",
    "refrigerator",
    "remote-control",
    "scarf",
    "sculpture",
    "shelf",
    "shoe",
    "shower-stall",
    "sink",
    "small-appliance",
    "sofa",
    "stair",
    "stool",
    "switch",
    "table",
    "table-runner",
    "tablet",
    "tissue-paper",
    "toilet",
    "toothbrush",
    "towel",
    "tv-screen",
    "tv-stand",
    "umbrella",
    "utensil-holder",
    "vase",
    "vent",
    "wall",
    "wall-cabinet",
    "wall-plug",
    "wardrobe",
    "window",
    "rug",
    "logo",
    "bag",
    "set-of-clothing",
]

REPLICA_SCENE_IDS = [
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
]

REPLICA_SCENE_IDS_ = [
    "room_0",
    "room_1",
    "room_2",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
]

REPLICA_EXISTING_CLASSES = [
    0, 3, 7, 8, 10, 11, 
    12, 13, 14, 15, 16, 
    17, 18, 19, 20, 22, 
    23, 26, 29, 31, 34, 
    35, 37, 40, 44, 47, 
    52, 54, 56, 59, 60, 
    61, 62, 63, 64, 65, 
    70, 71, 76, 78, 79, 
    80, 82, 83, 87, 88, 
    91, 92, 93, 95, 97, 
    98]

REPLICA_EXISTING_CLASSES_ROOM0 = [
    3, 11, 12, 13, 18, 
    19, 20, 29, 31, 37, 
    40, 44, 47, 59, 60, 
    63, 64, 65, 76, 78, 
    79, 80, 91, 92, 93, 
    95, 97, 98]

REPLICA_EXISTING_CLASSES_ROOM1 = [
    3, 7, 11, 12, 13, 
    18, 26, 31, 37, 40, 
    44, 47, 54, 56, 59, 
    61, 64, 79, 91, 92, 
    93, 95, 97, 98]

REPLICA_EXISTING_CLASSES_ROOM2 = [
    12, 14, 15, 16, 20,
    31, 37, 40, 44, 47,
    64, 70, 71, 79, 80,
    91, 92, 93, 95, 97, 
    98]
REPLICA_EXISTING_CLASSES_OFFICE0 = [
    10, 12, 17, 20, 22, 
    31, 35, 37, 40, 44, 
    47, 56, 60, 63, 76, 
    79, 80, 82, 83, 87, 
    92, 93, 95, 98
]
REPLICA_EXISTING_CLASSES_OFFICE1 = [
    10, 11, 12, 14, 20,
    23, 31, 34, 35, 37,
    40, 47, 52, 56, 60,
    61, 79, 80, 83, 92, 
    93, 95]
REPLICA_EXISTING_CLASSES_OFFICE2 = [
    10, 14, 17, 20, 22,
    29, 31, 37, 40, 47,
    56, 76, 78, 80, 82,
    83, 87, 92, 93, 95, 
    97]
REPLICA_EXISTING_CLASSES_OFFICE3 = [
    8, 10, 12, 14, 15,
    17, 20, 22, 29, 31,
    35, 37, 40, 47, 56,
    62, 76, 79, 80, 82, 
    83, 88, 92, 93, 95,
    97]
REPLICA_EXISTING_CLASSES_OFFICE4 = [
    8, 10, 17, 20, 22,
    31, 37, 40, 47, 56,
    80, 87, 92, 93, 95,
    97]

OVOSLAM_COLORED_LABELS = [
    "panel", "vase", "clock", "pot", "window", "bottle", "indoor-plant", "pillow", 
    "blinds", "lamp", "pillar", "wall-plug", "wall", "cushion", "switch", "picture", 
    "bench", "box", "shelf", "stool", "book", "cloth", "rug", "table", "monitor", 
    "ceiling", "bowl", "camera", "basket", "nightstand", "blanket", "door", "bed", 
    "comforter", "plate", "chair", "vent", "bin", "desk", "floor", "cabinet", 
    "sculpture", "tv-screen"
]

LABEL_TO_COLOR = {
    "panel":       (220, 20, 60),    # crimson
    "vase":        (255, 127, 0),    # orange
    "clock":       (255, 215, 0),    # gold
    "pot":         (0, 128, 0),      # green
    "window":      (0, 191, 255),    # deep sky blue
    "bottle":      (106, 90, 205),   # slate blue
    "indoor-plant":(34, 139, 34),    # forest green
    "pillow":      (255, 192, 203),  # pink
    "blinds":      (0, 255, 255),    # cyan
    "lamp":        (255, 0, 255),    # magenta
    "pillar":      (139, 69, 19),    # saddle brown
    "wall-plug":   (128, 0, 0),      # maroon
    "wall":        (128, 128, 128),  # gray
    "cushion":     (173, 216, 230),  # light blue
    "switch":      (210, 105, 30),   # chocolate
    "picture":     (244, 164, 96),   # sandy brown
    "bench":       (128, 128, 0),    # olive
    "box":         (154, 205, 50),   # yellow green
    "shelf":       (0, 100, 0),      # dark green
    "stool":       (70, 130, 180),   # steel blue
    "book":        (230, 230, 250),  # lavender
    "cloth":       (255, 160, 122),  # light salmon
    "rug":         (255, 228, 196),  # bisque
    "table":       (255, 69, 0),     # orange red
    "monitor":     (138, 43, 226),   # blue violet
    "ceiling":     (127, 255, 212),  # aquamarine
    "bowl":        (255, 140, 0),    # dark orange
    "camera":      (128, 0, 128),    # purple
    "basket":      (255, 99, 71),    # tomato
    "nightstand":  (165, 42, 42),    # brown
    "blanket":     (221, 160, 221),  # plum
    "door":        (139, 0, 0),      # dark red
    "bed":         (184, 134, 11),   # dark goldenrod
    "comforter":   (0, 128, 128),    # teal
    "plate":       (152, 251, 152),  # pale green
    "chair":       (178, 34, 34),    # firebrick
    "vent":        (112, 128, 144),  # slate gray
    "bin":         (0, 139, 139),    # dark cyan
    "desk":        (160, 82, 45),    # sienna
    "floor":       (188, 143, 143),  # rosy brown
    "cabinet":     (139, 0, 139),    # dark magenta
    "sculpture":   (218, 112, 214),  # orchid
    "tv-screen":   (199, 21, 133),   # medium violet red
}
