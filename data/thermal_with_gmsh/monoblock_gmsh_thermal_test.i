#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

end_time = 10
time_step = 2

#stressFreeTemp=20   # degC
coolantTemp=100     # degC
surfHeatFlux=10e6   # W/m^2

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Mesh]
  type = FileMesh
  file = 'monoblock_3d.msh'
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = SECOND
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
  [thermal_cucrzr]
    type = HeatConductionMaterial
    block = 'pipe-cucrzr'
    thermal_conductivity = 345.0
    specific_heat = 400.0
  []
  [thermal_cu]
    type = HeatConductionMaterial
    block = 'interlayer-cu'
    thermal_conductivity = 380.0
    specific_heat = 400.0
  []
  [thermal_w]
    type = HeatConductionMaterial
    block = 'armour-w'
    thermal_conductivity = 140.0
    specific_heat = 145.0
  []

  [density_cucrzr]
      type = GenericConstantMaterial
      block = 'pipe-cucrzr'
      prop_names = 'density'
      prop_values = 8800.0
  []
  [density_cu]
    type = GenericConstantMaterial
    block = 'interlayer-cu'
    prop_names = 'density'
    prop_values = 8800.0
  []
  [density_w]
    type = GenericConstantMaterial
    block = 'armour-w'
    prop_names = 'density'
    prop_values = 19150.0
  []

  [heat_transfer_coefficient]
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
    type = FunctionNeumannBC
    variable = temperature
    boundary = 'bc-top-heatflux'
    function = '${fparse surfHeatFlux}*(1-exp(-t))'
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
  type = Transient
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre    boomeramg'
  end_time = ${end_time}
  dt = ${time_step}
[]

[Postprocessors]
  [max_temp]
    type = ElementExtremeValue
    variable = temperature
  []
[]

[Outputs]
  exodus = true
[]
