#!/usr/bin/python3
"""
write_data_to_csv.

A script to generate csv files containing temperature-variant material
properties for monoblock materials.
Materials included are OFHC copper for the coolant pipe, copper-corundum-
zirconium (CuCrZr) for the interlayer, and tungsten for the armour.
The files are generated in the same directory as this script in csv format with
one file per property per material.

(c) UK Atomic Energy Authority, 2023
"""

import os
import csv

DATA_DIR = os.path.realpath(os.path.dirname(__file__))

# Thermomechanical properties for OFHC copper
# Temperature in degC
copper_temperature = [
    20,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
    1000,
]


# OFHC Copper - Conductivity in W.m^-1.K^-1
copper_conductivity = [
    401,
    398,
    395,
    391,
    388,
    384,
    381,
    378,
    374,
    371,
    367,
    364,
    360,
    357,
    354,
    350,
    347,
    344,
    340,
    337,
    334,
]

with open(f"{DATA_DIR}/copper_conductivity.csv", "w") as file:
    file.write("# Conductivity data for copper\n")
    file.write("# Temp. (C), Conductivity (W.m^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, cond in zip(copper_temperature, copper_conductivity):
        data = [temp, cond]
        writer.writerow(data)


# OFHC Copper - Specific Heat in J.kg^-1.K^-1
copper_specific_heat = [
    388,
    390,
    394,
    398,
    401,
    406,
    410,
    415,
    419,
    424,
    430,
    435,
    441,
    447,
    453,
    459,
    466,
    472,
    479,
    487,
    494,
]

with open(f"{DATA_DIR}/copper_specific_heat.csv", "w") as file:
    file.write("# Specific heat data for copper\n")
    file.write("# Temp. (C), Specific heat (J.kg^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, spec in zip(copper_temperature, copper_specific_heat):
        data = [temp, spec]
        writer.writerow(data)


# OFHC Copper - Elastic modulus in Pa
copper_elastic_modulus = [
    117e9,
    116e9,
    114e9,
    112e9,
    110e9,
    108e9,
    105e9,
    102e9,
    98e9,
]

with open(f"{DATA_DIR}/copper_elastic_modulus.csv", "w") as file:
    file.write("# Elastic modulus data for copper\n")
    file.write("# Temp. (C), Elastic Modulus (Pa)\n")

    writer = csv.writer(file)
    for temp, elas in zip(copper_temperature, copper_elastic_modulus):
        data = [temp, elas]
        writer.writerow(data)


# OFHC Copper - Poisson's ratio
copper_poissons_ratio = [
    0.33,
]


# OFHC Copper - Density in kg.m^-3
copper_density = [
    8940,
    8926,
    8903,
    8879,
    8854,
    8829,
    8802,
    8774,
    8744,
    8713,
    8681,
    8647,
    8612,
    8575,
    8536,
    8495,
    8453,
    8409,
    8363,
]

with open(f"{DATA_DIR}/copper_density.csv", "w") as file:
    file.write("# Density data for copper\n")
    file.write("# Temp. (C), Density (kg.m^-3)\n")

    writer = csv.writer(file)
    for temp, dens in zip(copper_temperature, copper_density):
        data = [temp, dens]
        writer.writerow(data)


# OFHC Copper - Mean or Secant CTE
# Reference temperature 20 degC
copper_cte = [
    16.7e-6,
    17.0e-6,
    17.2e-6,
    17.5e-6,
    17.7e-6,
    17.8e-6,
    18.0e-6,
    18.1e-6,
    18.2e-6,
    18.4e-6,
    18.5e-6,
    18.7e-6,
    18.8e-6,
    19.0e-6,
    19.1e-6,
    19.3e-6,
    19.6e-6,
    19.8e-6,
    20.1e-6,
]

with open(f"{DATA_DIR}/copper_cte.csv", "w") as file:
    file.write("# Thermal expansion data for copper\n")
    file.write("# Temp. (C), CTE\n")

    writer = csv.writer(file)
    for temp, cte in zip(copper_temperature, copper_cte):
        data = [temp, cte]
        writer.writerow(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Thermomechanical properties for Copper-Chromium-Zirconium # (Cu-Cr-Zr)
# Temperature in degC
cucrzr_temperature = [
    20,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
]


# Cu-Cr-Zr - Conductivity in W.m^-1.K^-1
cucrzr_conductivity = [
    318,
    324,
    333,
    339,
    343,
    345,
    346,
    347,
    347,
    346,
    346,
]

with open(f"{DATA_DIR}/cucrzr_conductivity.csv", "w") as file:
    file.write("# Conductivity data for Cu-Cr-Zr\n")
    file.write("# Temp. (C), Conductivity (W.m^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, cond in zip(cucrzr_temperature, cucrzr_conductivity):
        data = [temp, cond]
        writer.writerow(data)


# Cu-Cr-Zr  - Specific Heat in J.kg^-1.K^-1
cucrzr_specific_heat = [
    390,
    393,
    398,
    402,
    407,
    412,
    417,
    422,
    427,
    432,
    437,
    442,
    447,
    452,
    458,
]

with open(f"{DATA_DIR}/cucrzr_specific_heat.csv", "w") as file:
    file.write("# Specific heat data for Cu-Cr-Zr\n")
    file.write("# Temp. (C), Specific heat (J.kg^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, spec in zip(cucrzr_temperature, cucrzr_specific_heat):
        data = [temp, spec]
        writer.writerow(data)


# Cu-Cr-Zr - Elastic modulus in Pa
cucrzr_elastic_modulus = [
    128e9,
    127e9,
    127e9,
    125e9,
    123e9,
    121e9,
    118e9,
    116e9,
    113e9,
    110e9,
    106e9,
    100e9,
    95e9,
    90e9,
    86e9,
]

with open(f"{DATA_DIR}/cucrzr_elastic_modulus.csv", "w") as file:
    file.write("# Elastic modulus data for Cu-Cr-Zr\n")
    file.write("# Temp. (C), Elastic Modulus (Pa)\n")

    writer = csv.writer(file)
    for temp, elas in zip(cucrzr_temperature, cucrzr_elastic_modulus):
        data = [temp, elas]
        writer.writerow(data)


# Cu-Cr-Zr - Poisson's ratio
cucrzr_poissons_ratio = [
    0.33,
]


# Cu-Cr-Zr - Density in kg.m^-3
cucrzr_density = [
    8900,
    8886,
    8863,
    8840,
    8816,
    8791,
    8797,
    8742,
    8716,
    8691,
    8665,
]

with open(f"{DATA_DIR}/cucrzr_density.csv", "w") as file:
    file.write("# Density data for Cu-Cr-Zr\n")
    file.write("# Temp. (C), Density (kg.m^-3)\n")

    writer = csv.writer(file)
    for temp, dens in zip(cucrzr_temperature, cucrzr_density):
        data = [temp, dens]
        writer.writerow(data)


# Cu-Cr-Zr - Mean or Secant CTE
# Reference temperature 20 degC
cucrzr_cte = [
    16.7e-6,
    17.0e-6,
    17.3e-6,
    17.5e-6,
    17.7e-6,
    17.8e-6,
    18.0e-6,
    18.0e-6,
    18.1e-6,
    18.2e-6,
    18.4e-6,
    18.5e-6,
    18.6e-6,
]

with open(f"{DATA_DIR}/cucrzr_cte.csv", "w") as file:
    file.write("# Thermal expansion data for Cu-Cr-Zr\n")
    file.write("# Temp. (C), CTE\n")

    writer = csv.writer(file)
    for temp, cte in zip(cucrzr_temperature, cucrzr_cte):
        data = [temp, cte]
        writer.writerow(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Thermomechanical properties for Tungsten
# Temperature in degC
tungsten_temperature = [
    20,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
    1000,
    1100,
    1200,
]


# TUNGSTEN - Conductivity in W.m^-1.K^-1
tungsten_conductivity = [
    173,
    170,
    165,
    160,
    156,
    151,
    147,
    143,
    140,
    136,
    133,
    130,
    127,
    125,
    122,
    120,
    118,
    116,
    114,
    112,
    110,
    108,
    105,
]

with open(f"{DATA_DIR}/tungsten_conductivity.csv", "w") as file:
    file.write("# Conductivity data for tungsten\n")
    file.write("# Temp. (C), Conductivity (W.m^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, cond in zip(tungsten_temperature, tungsten_conductivity):
        data = [temp, cond]
        writer.writerow(data)


# TUNGSTEN  - Specific Heat in J.kg^-1.K^-1
tungsten_specific_heat = [
    129,
    130,
    132,
    133,
    135,
    136,
    138,
    139,
    141,
    142,
    144,
    145,
    147,
    148,
    150,
    151,
    152,
    154,
    155,
    156,
    158,
    160,
    163,
]

with open(f"{DATA_DIR}/tungsten_specific_heat.csv", "w") as file:
    file.write("# Specific heat data for tungsten\n")
    file.write("# Temp. (C), Specific heat (J.kg^-1.K^-1)\n")

    writer = csv.writer(file)
    for temp, spec in zip(tungsten_temperature, tungsten_specific_heat):
        data = [temp, spec]
        writer.writerow(data)


# Tungsten - Elastic modulus in Pa
tungsten_elastic_modulus = [
    398e9,
    398e9,
    397e9,
    397e9,
    396e9,
    396e9,
    395e9,
    394e9,
    393e9,
    391e9,
    390e9,
    388e9,
    387e9,
    385e9,
    383e9,
    381e9,
    379e9,
    376e9,
    374e9,
    371e9,
    368e9,
    362e9,
    356e9,
]

with open(f"{DATA_DIR}/tungsten_elastic_modulus.csv", "w") as file:
    file.write("# Elastic modulus data for tungsten\n")
    file.write("# Temp. (C), Elastic Modulus (Pa)\n")

    writer = csv.writer(file)
    for temp, elas in zip(tungsten_temperature, tungsten_elastic_modulus):
        data = [temp, elas]
        writer.writerow(data)


# TUNGSTEN - Poisson's ratio
tungsten_poissons_ratio = [
    0.29,
]


# TUNGSTEN - Density in kg.m^-3
tungsten_density = [
    19300,
    19290,
    19280,
    19270,
    19250,
    19240,
    19230,
    19220,
    19200,
    19190,
    19180,
    19170,
    19150,
    19140,
    19130,
    19110,
    19100,
    19080,
    19070,
    19060,
    19040,
    19010,
    18990,
]

with open(f"{DATA_DIR}/tungsten_density.csv", "w") as file:
    file.write("# Density data for tungsten\n")
    file.write("# Temp. (C), Density (kg.m^-3)\n")

    writer = csv.writer(file)
    for temp, dens in zip(tungsten_temperature, tungsten_density):
        data = [temp, dens]
        writer.writerow(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Thermal expansion data for Tungsten
# Temperature
tungsten_temperature_cte = [
    20,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
    3200,
]

# Tungsten - Mean or Secant CTE
# Reference temperature 20 degC
tungsten_cte = [
    4.50e-6,
    4.50e-6,
    4.53e-6,
    4.58e-6,
    4.63e-6,
    4.68e-6,
    4.72e-6,
    4.76e-6,
    4.81e-6,
    4.85e-6,
    4.89e-6,
    4.98e-6,
    5.08e-6,
    5.18e-6,
    5.30e-6,
    5.43e-6,
    5.57e-6,
    5.74e-6,
    5.93e-6,
    6.15e-6,
    6.40e-6,
    6.67e-6,
]

with open(f"{DATA_DIR}/tungsten_cte.csv", "w") as file:
    file.write("# Thermal expansion data for tungsten\n")
    file.write("# Temp. (C), CTE\n")

    writer = csv.writer(file)
    for temp, cte in zip(tungsten_temperature_cte, tungsten_cte):
        data = [temp, cte]
        writer.writerow(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Heat transfer coefficient data for Water
# Temperature
water_temperature = [
    1,
    100,
    150,
    200,
    250,
    295,
]
water_htc = [
    4,
    109.1e3,
    115.9e3,
    121.01e3,
    128.8e3,
    208.2e3,
]

with open(f"{DATA_DIR}/water_htc.csv", "w") as file:
    file.write("# Heat transfer coefficient data for water\n")
    file.write("# Temp. (C), HTC (W.m^-2.K^-1)\n")

    writer = csv.writer(file)
    for temp, htc in zip(water_temperature, water_htc):
        data = [temp, htc]
        writer.writerow(data)