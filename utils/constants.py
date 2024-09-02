from os.path import dirname, abspath

PROJECT_ROOT = dirname(dirname(abspath(__file__)))

NON_OVERLAPPING_SDGS = {
    "SDG01": ["SDG07", "SDG09", "SDG14", "SDG15"],
    "SDG02": ["SDG07", "SDG09", "SDG11", "SDG17"],
    "SDG03": ["SDG07", "SDG09", "SDG13", "SDG14"],
    "SDG04": ["SDG06", "SDG07", "SDG9", "SDG14"],
    "SDG05": ["SDG06", "SDG07", "SDG09", "SDG14"],
    "SDG06": ["SDG07", "SDG09", "SDG16", "SDG17"],
    "SDG07": ["SDG05", "SDG10", "SDG12", "SDG16"],
    "SDG08": ["SDG06", "SDG07", "SDG12", "SDG13"],
    "SDG09": ["SDG04", "SDG06", "SDG14", "SDG17"],
    "SDG10": ["SDG07", "SDG09", "SDG13", "SDG14"],
    "SDG11": ["SDG02", "SDG07", "SDG15", "SDG17"],
    "SDG12": ["SDG07", "SDG08", "SDG9", "SDG10"],
    "SDG13": ["SDG03", "SDG08", "SDG10", "SDG16"],
    "SDG14": ["SDG04", "SDG05", "SDG09", "SDG10"],
    "SDG15": ["SDG01", "SDG07", "SDG11", "SDG16"],
    "SDG16": ["SDG06", "SDG07", "SDG13", "SDG14"],
    "SDG17": ["SDG02", "SDG06", "SDG09", "SDG11"]
}