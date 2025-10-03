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

    def get_targets(self):
        target = {
            'age' : -1,
            'other_type' : -1,
            'cretaceous_type' : -1,
            'shale_type' : -1,
            'cretaceous_other_type' : -1,
            'paleozoic_type' : -1,
            'precambrian_type' : -1,
            'crystalline_type' : -1,
            'light_type' : -1,
            'dark_type' : -1,
            'red_type' : -1
        }

        unit = self

        while unit:
            target[unit.category] = unit.label
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
Cretaceous_Mineral_Other = Grain('Cretaceous Mineral Other', 6, 'createceous_other_type', parent=Non_Shale) #TODO: Ask about this one

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
    'chert' : Chert,
    'unknown' : Unknown,
    'secondary' : Secondary,
    'gypsum' : Gypsum,

}

class GrainLabel:
    SHALE = 'shale'
    FELSIC = 'felsic'

CLASS_MAP = {
    1 : GrainLabel.SHALE,
    0 : GrainLabel.FELSIC,
}