#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,1mat,thermal,steady,
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# NOTE: only used for transient solves
endTime = 30
timeStep = 0.5

# Geometric Properties
lengX = 100e-3  # m
lengY = 50e-3   # m

# Mesh Properties
nElemX = 10
nElemY = 5
eType = QUAD4 # QUAD4 for 1st order, QUAD8 for 2nd order

# Thermal Loads/BCs
coolantTemp = 20.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1
surfHeatFlux = 500.0e3    # W.m^-2
timeConst = 1   # s

# Material Properties: Pure (OFHC) Copper at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 2
        nx = ${nElemX}
        ny = ${nElemY}
        xmax = ${lengX}
        ymax = ${lengY}
        elem_type = ${eType}
    []
[]

[Variables]
    [temperature]
        initial_condition = ${coolantTemp}
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = temperature
    []
[]

[Materials]
    [copper_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = ${cuThermCond}
        specific_heat = ${cuSpecHeat}
    []
    [copper_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${cuDensity}
    []
[]

[BCs]
    [heat_flux_out]
        type = ConvectiveHeatFluxBC
        variable = temperature
        boundary = 'left'
        T_infinity = ${coolantTemp}
        heat_transfer_coefficient = ${heatTransCoeff}
    []
    [heat_flux_in]
        type = FunctionNeumannBC
        variable = temperature
        boundary = 'right'
        function = '${fparse surfHeatFlux}*(1-exp(-(1/${timeConst})*t))'
    []
[]

[Executioner]
    type = Transient
    end_time= ${endTime}
    dt = ${timeStep}
[]

[Postprocessors]
    [max_temp]
        type = NodalExtremeValue
        variable = temperature
    []
    [avg_temp]
        type = AverageNodalVariableValue
        variable = temperature
    []
[]

[Outputs]
    exodus = true
[]