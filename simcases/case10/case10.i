#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,1mat,thermomechanical,steady,
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

#endTime= 1
#timeStep = 1

# Thermal Loads/BCs
coolantTemp = 100.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1
surfHeatFlux = 5.0e6    # W.m^-2

# Material Properties:
# Thermal Props:OFHC) Copper at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

# Mechanical Props: OFHC Copper 250degC
cuEMod = 108e9       # Pa
cuPRatio = 0.33      # -

# Thermo-mechanical coupling
stressFreeTemp = 20 # degC
cuThermExp = 17.8e-6 # 1/degC

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------


[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'case10.msh'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = FIRST
        initial_condition = ${coolantTemp}
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    #[time_derivative]
    #    type = HeatConductionTimeDerivative
    #    variable = temperature
    #[]
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = true
        #material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
        #material_output_order = FIRST       # CONSTANT, FIRST, SECOND,
        strain = SMALL                     # SMALL or FINITE
        automatic_eigenstrain_names = true
        generate_output = 'vonmises_stress stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy'
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
    [copper_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${cuEMod}
        poissons_ratio = ${cuPRatio}
    []
    [copper_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${cuThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
    []

    [stress]
        type = ComputeLinearElasticStress # ComputeLinearElasticStress or ComputeFiniteStrainElasticStress
    []
[]

[BCs]
    [heat_flux_out]
        type = ConvectiveHeatFluxBC
        variable = temperature
        boundary = 'bc-pipe-htc'
        T_infinity = ${coolantTemp}
        heat_transfer_coefficient = ${heatTransCoeff}
    []
    [heat_flux_in]
        type = NeumannBC
        variable = temperature
        boundary = 'bc-top-heatflux'
        value = ${surfHeatFlux}
    []

    [mech_dispx_bc]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-base-disp'
        value = 0
    []
    [mech_dispy_bc]
        type = DirichletBC
        variable = disp_y
        boundary = 'bc-base-disp'
        value = 0
    []
    [mech_dispz_bc]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-base-disp'
        value = 0
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
    #end_time= ${endTime}
    #dt = ${timeStep}
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

    [max_x_disp]
        type = NodalExtremeValue
        variable = disp_x
    []
    [max_xx_strain]
        type = ElementExtremeValue
        variable = strain_xx
    []
    [avg_xx_strain]
        type = ElementAverageValue
        variable = strain_yy
    []
[]

[Outputs]
    exodus = true
[]