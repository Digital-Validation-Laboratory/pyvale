#-------------------------------------------------------------------------
# pyvale: gmsh,monoblock,3mat,thermal,steady,
#-------------------------------------------------------------------------

orderString=FIRST

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loads and BCs
stressFreeTemp=20   # degC
coolantTemp=150     # degC
surfHeatFlux=10e6   # W/m^2

# mesh file
# Mesh file string
mesh_file = 'case18.msh'

#-------------------------------------------------------------------------


[Mesh]
  type = FileMesh
  file = ${mesh_file}
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = ${orderString}
    initial_condition = ${coolantTemp}
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
[]

[Functions]
  [cucrzr_thermal_expansion_func]
    type = PiecewiseLinear
    data_file = ./data/cucrzr_cte.csv
    format = columns
  []
  [copper_thermal_expansion_func]
    type = PiecewiseLinear
    data_file = ./data/copper_cte.csv
    format = columns
  []
  [tungsten_thermal_expansion_func]
    type = PiecewiseLinear
    data_file = ./data/tungsten_cte.csv
    format = columns
  []

  [cucrzr_thermal_conductivity_func]
    type = PiecewiseLinear
    data_file = ./data/cucrzr_conductivity.csv
    format = columns
  []
  [copper_thermal_conductivity_func]
    type = PiecewiseLinear
    data_file = ./data/copper_conductivity.csv
    format = columns
  []
  [tungsten_thermal_conductivity_func]
    type = PiecewiseLinear
    data_file = ./data/tungsten_conductivity.csv
    format = columns
  []

  [cucrzr_density_func]
    type = PiecewiseLinear
    data_file = ./data/cucrzr_density.csv
    format = columns
  []
  [copper_density_func]
    type = PiecewiseLinear
    data_file = ./data/copper_density.csv
    format = columns
  []
  [tungsten_density_func]
    type = PiecewiseLinear
    data_file = ./data/tungsten_density.csv
    format = columns
  []

  [cucrzr_elastic_modulus_func]
    type = PiecewiseLinear
    data_file = ./data/cucrzr_elastic_modulus.csv
    format = columns
  []
  [copper_elastic_modulus_func]
    type = PiecewiseLinear
    data_file = ./data/copper_elastic_modulus.csv
    format = columns
  []
  [tungsten_elastic_modulus_func]
    type = PiecewiseLinear
    data_file = ./data/tungsten_elastic_modulus.csv
    format = columns
  []

  [cucrzr_specific_heat_func]
    type = PiecewiseLinear
    data_file = ./data/cucrzr_specific_heat.csv
    format = columns
  []
  [copper_specific_heat_func]
    type = PiecewiseLinear
    data_file = ./data/copper_specific_heat.csv
    format = columns
  []
  [tungsten_specific_heat_func]
    type = PiecewiseLinear
    data_file = ./data/tungsten_specific_heat.csv
    format = columns
  []
[]

[Materials]
  [cucrzr_thermal_conductivity]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = thermal_conductivity
    function = cucrzr_thermal_conductivity_func
    block = 'pipe-cucrzr'
  []
  [copper_thermal_conductivity]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = thermal_conductivity
    function = copper_thermal_conductivity_func
    block = 'interlayer-cu'
  []
  [tungsten_thermal_conductivity]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = thermal_conductivity
    function = tungsten_thermal_conductivity_func
    block = 'armour-w'
  []

  [cucrzr_density]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = density
    function = cucrzr_density_func
    block = 'pipe-cucrzr'
  []
  [copper_density]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = density
    function = copper_density_func
    block = 'interlayer-cu'
  []
  [tungsten_density]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = density
    function = tungsten_density_func
    block = 'armour-w'
  []

  [cucrzr_elastic_modulus]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = elastic_modulus
    function = cucrzr_elastic_modulus_func
    block = 'pipe-cucrzr'
  []
  [copper_elastic_modulus]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = elastic_modulus
    function = copper_elastic_modulus_func
    block = 'interlayer-cu'
  []
  [tungsten_elastic_modulus]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = elastic_modulus
    function = tungsten_elastic_modulus_func
    block = 'armour-w'
  []

  [cucrzr_specific_heat]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = specific_heat
    function = cucrzr_specific_heat_func
    block = 'pipe-cucrzr'
  []
  [copper_specific_heat]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = specific_heat
    function = copper_specific_heat_func
    block = 'interlayer-cu'
  []
  [tungsten_specific_heat]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = specific_heat
    function = tungsten_specific_heat_func
    block = 'armour-w'
  []

  [cucrzr_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.33
    block = 'pipe-cucrzr'
  []
  [copper_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.33
    block = 'interlayer-cu'
  []
  [tungsten_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.29
    block = 'armour-w'
  []

  [cucrzr_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = cucrzr_thermal_expansion_func
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'pipe-cucrzr'
  []
  [copper_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = copper_thermal_expansion_func
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'interlayer-cu'
  []
  [tungsten_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = tungsten_thermal_expansion_func
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'armour-w'
  []

  [coolant_heat_transfer_coefficient]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      1 4
      100 109.1e3
      150 115.9e3
      200 121.01e3
      250 128.8e3
      295 208.2e3
    '
    variable = temperature
    property = heat_transfer_coefficient
    boundary = 'bc-pipe-heattransf'
  []
[]

[BCs]
  [heat_flux_in]
    type = NeumannBC
    variable = temperature
    boundary = 'bc-top-heatflux'
    value = ${surfHeatFlux}
  []
  [heat_flux_out]
    type = ConvectiveHeatFluxBC
    variable = temperature
    boundary = 'bc-pipe-heattransf'
    T_infinity = ${coolantTemp}
    heat_transfer_coefficient = heat_transfer_coefficient
  []
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre    boomeramg'
[]

[Postprocessors]
    [temp_max]
      type = NodalExtremeValue
      variable = temperature
  []
  [temp_avg]
      type = AverageNodalVariableValue
      variable = temperature
  []
[]

[Outputs]
  exodus = true
[]