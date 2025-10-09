import torch

class Grain:
    def __init__(self, name, label, category, parent=None):
        self.name = name
        self.label = label
        self.category = category
        self.parent = parent

    def get_lineage(self):
        lineage = []
        unit = self

        while unit:
            lineage.append(unit)
            unit = unit.parent

        return lineage

    def get_target(self):
        target = {
            'age' : torch.tensor(-1),
            'other_type' : torch.tensor(-1),
            'cretaceous_type' : torch.tensor(-1),
            'shale_type' : torch.tensor(-1),
            'cretaceous_other_type' : torch.tensor(-1),
            'paleozoic_type' : torch.tensor(-1),
            'precambrian_type' : torch.tensor(-1),
            'crystalline_type' : torch.tensor(-1),
            'light_type' : torch.tensor(-1),
            'dark_type' : torch.tensor(-1),
            'red_type' : torch.tensor(-1)
        }

        unit = self

        while unit:
            target[unit.category] = torch.tensor(unit.label)
            unit = unit.parent

        return target

Other = Grain('Other', 0, 'age')
Cretaceous = Grain('Cretaceous', 1, 'age')
Paleozoic = Grain('Paleozoic', 2, 'age')
Precambrian = Grain('Precambrian', 3, 'age')

Chert = Grain('Chert', 0, 'other_type', parent=Other)
Unknown = Grain('Unknown', 1, 'other_type',parent=Other)
Secondary = Grain('Secondary', 2, 'other_type',parent=Other)
Gypsum = Grain('Gypsum', 3, 'other_type',parent=Other)

Shale = Grain('Shale', 0, 'cretaceous_type', parent=Cretaceous)
Non_Shale = Grain('Cretaceous Non-Shale', 1, 'cretaceous_type',parent=Cretaceous)
Gray_Shale = Grain('Gray Shale', 0, 'shale_type', parent=Shale)
Speckled_Shale = Grain('Speckled Shale', 1, 'shale_type', parent=Shale)
Dark_Limestone = Grain('Dark Limestone', 0, 'createceous_other_type', parent=Non_Shale)
Inoceramus = Grain('Inoceramus', 1, 'createceous_other_type', parent=Non_Shale)
Cretaceous_Other = Grain('Cretaceous Other', 2, 'createceous_other_type', parent=Non_Shale)
Pyrite = Grain('Pyrite', 3, 'createceous_other_type', parent=Non_Shale)
Lignite = Grain('Lignite', 4, 'createceous_other_type', parent=Non_Shale)
Ostrander_Sand = Grain('Ostrander Sand', 5, 'createceous_other_type', parent=Non_Shale)

Carbonate = Grain('Carbonate', 0, 'paleozoic_type', parent=Paleozoic)
Paleozoic_Other = Grain('Paleozoic Other', 1, 'paleozoic_type', parent=Paleozoic)

Crystalline = Grain('Crystalline', 0, 'precambrian_type', parent=Precambrian)
Precambrian_Other = Grain('Precambrian Other', 1, 'precambrian_type', parent=Precambrian)
Light = Grain('Light', 0, 'crystalline_type', parent=Crystalline)
Dark = Grain('Dark', 1, 'crystalline_type', parent=Crystalline)
Red = Grain('Red', 2, 'crystalline_type', parent=Crystalline)
Felsic = Grain('Felsic', 0, 'light_type', parent=Light)
Quartzite = Grain('Quartzite', 1, 'light_type', parent=Light)
Clear_Quartz = Grain('Clear Quartz', 2, 'light_type', parent=Light)
Mafic_Igneous = Grain('Mafic Igneous', 0, 'dark_type', parent=Dark)
Metased_Volcanic = Grain('Metased/Volcanic', 1, 'dark_type', parent=Dark)
Iron_Formation = Grain('Iron Formation', 0, 'red_type', parent=Red)
Red_Volcanic = Grain('Red Volcanic', 1, 'red_type', parent=Red)
Arkosic = Grain('Arkosic', 2, 'red_type', parent=Red)
Quartz_Arenite = Grain('Quartz Arenite', 'red_type', 3, parent=Red)

GRAIN_MAP = {
    'other' : Other,
    'cretaceous' : Cretaceous,
    'paleozoic' : Paleozoic,
    'precambrian' : Precambrian,

    'chert' : Chert,
    'unknown' : Unknown,
    'secondary' : Secondary,
    'gypsum' : Gypsum,

    'shale' : Shale,
    'non_shale' : Shale,
    'gray_shale' : Gray_Shale,
    'speckled_shale' : Speckled_Shale,
    'dark_limestone' : Dark_Limestone,
    'inoceramus' : Inoceramus,
    'cretaceous_other' : Cretaceous_Other,
    'pyrite' : Pyrite,
    'lignite' : Lignite,
    'ostrander_sand' : Ostrander_Sand,

    'carbonate' : Carbonate,
    'paleozoic_other' : Paleozoic_Other,

    'crystalline' : Crystalline,
    'precambrian_other' : Precambrian_Other,
    'light' : Light,
    'dark' : Dark,
    'red' : Red,
    'felsic' : Felsic,
    'quartzite' : Quartzite,
    'clear_quartz' : Clear_Quartz,
    'mafic_igneous' : Mafic_Igneous,
    'metased_volcanic' : Metased_Volcanic,
    'iron_formation' : Iron_Formation,
    'red_volcanic' : Red_Volcanic,
    'arkosic' : Arkosic,
    'quartz_arenite' : Quartz_Arenite,
}

def targets_to_label(output, confidence, threshold : float = .5):
    if confidence['age'] < threshold:
        return Unknown.name, confidence['age']

    match output['age']:
        case 0:
            if confidence['other_type'] > threshold:
                other_type_dict = {0 : Chert, 1 : Unknown, 2 : Secondary, 3 : Gypsum}
                return other_type_dict[output['other_type']].name, confidence['other_type']
            return Other.name, confidence['age']
        case 1:
            if confidence['cretaceous_type'] > threshold:
                match output['cretaceous_type']:
                    case 0:
                        if confidence['shale_type'] > threshold:
                            shale_type_dict = {0 : Gray_Shale, 1 : Speckled_Shale}
                            return shale_type_dict[output['shale_type']].name, confidence['shale_type']

                        return Shale.name, confidence['cretaceous_type']
                    case 1:
                        if confidence['cretaceous_other_type'] >  threshold:
                            non_shale_type_dict = {0 : Dark_Limestone, 1 : Inoceramus, 2 : Cretaceous_Other, 3 : Pyrite,
                                                   4 : Lignite, 5 : Ostrander_Sand}
                            return non_shale_type_dict[output['cretaceous_other_type']].name, confidence['cretaceous_other_type']
            return Cretaceous.name, confidence['age']
        case 2:
            if confidence['paleozoic_type'] > threshold:
                paleozoic_type_dict = {0 : Carbonate, 1 : Paleozoic_Other}
                return paleozoic_type_dict[output['paleozoic_type']].name, confidence['paleozoic_type']
            return Paleozoic.name, confidence['age']
        case 3:
            if confidence['precambrian_type'] > threshold:
                match output['precambrian_type']:
                    case 0:
                        if confidence['crystalline_type'] > threshold:
                            match output['crystalline_type']:
                                case 0:
                                    if confidence['light_type'] > threshold:
                                        light_type_dict = {0 : Felsic, 1 : Quartzite, 2 : Clear_Quartz}
                                        return light_type_dict[output['light_type']].name, confidence['light_type']
                                    return Light.name, confidence['crystalline_type']
                                case 1:
                                    if confidence['dark_type'] > threshold:
                                        dark_type_dict = {0 : Mafic_Igneous, 1 : Metased_Volcanic}
                                        return dark_type_dict[output['dark_type']].name, confidence['dark_type']
                                    return Dark.name, confidence['crystalline_type']
                                case 2:
                                    if confidence['red_type'] > threshold:
                                        red_type_dict = {0 : Iron_Formation, 1 : Red_Volcanic, 2 : Arkosic, 3 : Quartz_Arenite}
                                        return red_type_dict[output['red_type']].name, confidence['red_type']
                                    return Red.name, confidence['crystalline_type']
                        return Crystalline.name, confidence['precambrian_type']
                    case 1:
                        return Precambrian_Other.name, confidence['precambrian_type']
            return Precambrian.name, confidence['age']
    return Unknown.name, confidence['age']

#TODO: Get rid of
class GrainLabel:
    SHALE = 'shale'
    FELSIC = 'felsic'

CLASS_MAP = {
    1 : GrainLabel.SHALE,
    0 : GrainLabel.FELSIC,
}